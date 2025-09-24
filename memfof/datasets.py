# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

from memfof.utils.flow_utils import merge_flows
import numpy as np
import torch
import torch.utils.data as data

import os
import json
import random
from glob import glob
import os.path as osp
from memfof.utils import frame_utils
from memfof.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from memfof.utils.flow_utils import fill_invalid

from functools import reduce
from queue import Queue


def js_read(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


# Placeholder class, may contain:
# - images
# - flows
# - flow_masks
class SceneData:
    def __init__(self):
        self.images = {}
        self.flows = {}
        self.flow_masks = {}
        self.flow_mults = {}
        self.name = ""
        pass


class DataBlob:
    def __init__(self):
        pass


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, scene_params={}):
        self.augmentor = None
        self.sparse = sparse
        self.dataset = "unknown"
        self.subsample_groundtruth = False
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False

        self.scenes = []
        self.scene_params = scene_params

    @staticmethod
    def _flow_bfs(flows, s, t, max_depth):
        q = Queue()
        q.put(s)

        prev = {s: None}
        depth = {s: 0}

        ok = False
        while not ok and not q.empty():
            v = q.get()
            if depth[v] > max_depth or v not in flows:
                continue

            for u in flows[v]:
                if u not in prev:
                    prev[u] = v
                    depth[u] = depth[v] + 1
                    q.put(u)

                if u == t:
                    ok = True
                    break

        if not ok:
            return None

        res = []
        while t != s:
            res.append([prev[t], t])
            t = prev[t]

        res.reverse()
        return res

    # Default parameters are for the two frame FW flow case
    def process_scenes(
        self,
        frames=[(0, "left"), (1, "left")],
        flows=[((0, "left"), (1, "left"))],  # (source image, target image)
        sequence_bounds=(0, 1),  # range(min_image + bound[0], max_image + bound[1])
        invalid_images="clip",  # skip or clip
        invalid_flows="merge",
    ):  # skip or merge
        self.data = []
        for scene in self.scenes:  # `scene` is a SceneData class?
            if not scene.images:
                continue

            min_image = min(scene.images.keys())[0]
            max_image = max(scene.images.keys())[0]

            if sequence_bounds is None:
                i_list = sorted(set((x[0] for x in scene.images.keys())))
            else:
                i_list = list(
                    range(
                        min_image + sequence_bounds[0], max_image + sequence_bounds[1]
                    )
                )

            for i in i_list:
                valid_data = True

                image_list = []
                for di, cam in frames:
                    image = (i + di, cam)

                    if image not in scene.images:
                        if invalid_images == "skip":
                            valid_data = False
                        elif invalid_images == "clip":
                            image = (max(min_image, min(max_image, image[0])), image[1])
                        else:
                            raise Exception("Invalid image mode")

                    if image not in scene.images:
                        valid_data = False

                    if not valid_data:
                        break
                    else:
                        image_list.append(scene.images[image])

                (
                    flow_list,
                    flow_mask_list,
                    flow_mults_list,
                ) = [], [], []
                for img_pair in flows:
                    if img_pair is None:
                        flow_list.append(None)
                        flow_mask_list.append(None)
                        flow_mults_list.append(None)
                        continue

                    img1_, img2_ = img_pair

                    img1 = (img1_[0] + i, img1_[1])
                    img2 = (img2_[0] + i, img2_[1])

                    img_pairs = []
                    if (img1 not in scene.flows) or (img2 not in scene.flows[img1]):
                        if invalid_flows == "skip":
                            valid_data = False
                        elif invalid_flows == "merge":
                            img_pairs = FlowDataset._flow_bfs(
                                scene.flows,
                                img1,
                                img2,
                                max_depth=abs(img1[0] - img2[0]) + 10,
                            )

                            if img_pairs is None:
                                valid_data = False
                        else:
                            raise Exception("Invalid flow mode")
                    else:
                        img_pairs = [(img1, img2)]

                    if not valid_data:
                        break
                    else:
                        flow_list.append(
                            [scene.flows[img1_][img2_] for img1_, img2_ in img_pairs]
                        )

                    try:
                        flow_mask_list.append(
                            [
                                scene.flow_masks[img1_][img2_]
                                for img1_, img2_ in img_pairs
                            ]
                        )
                    except:
                        flow_mask_list.append(None)

                    try:
                        flow_mults_list.append(
                            [
                                scene.flow_mults[img1_][img2_]
                                for img1_, img2_ in img_pairs
                            ]
                        )
                    except:
                        flow_mults_list.append(None)

                if not valid_data:
                    continue

                new_data = DataBlob()
                new_data.images = image_list
                new_data.flows = flow_list
                new_data.flow_masks = flow_mask_list
                new_data.flow_mults = flow_mults_list
                new_data.extra_info = (scene.name, i, len(self.data), frames, flows)

                self.data.append(new_data)

    def __getitem__(self, index):
        while True:
            try:
                return self.fetch(index)
            except Exception as e:
                index = random.randint(0, len(self) - 1)
                raise e

    def fetch(self, index):
        if self.is_test:
            imgs = []
            for image in self.data[index].images:
                img = frame_utils.read_gen(image)
                img = np.array(img).astype(np.uint8)[..., :3]
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                imgs.append(img)

            return torch.stack(imgs), self.data[index].extra_info

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.data)

        imgs = []
        for image in self.data[index].images:
            img = frame_utils.read_gen(image)
            imgs.append(img)

        imgs = [np.array(img).astype(np.uint8) for img in imgs]

        flows, valids = [], []
        for flow_ind in range(len(self.data[index].flows)):
            if self.data[index].flows[flow_ind] is None:
                if len(imgs) > 0:
                    n, m = imgs[0].shape[:2]
                else:
                    n, m = 1

                flows.append(np.zeros((n, m, 2)).astype(np.float32))
                valids.append(np.zeros((n, m)).astype(np.float32))
                continue

            cur_flows, cur_valids = [], []

            if self.sparse:
                if self.dataset == "TartanAir":
                    for f, fm in zip(
                        self.data[index].flows[flow_ind],
                        self.data[index].flow_masks[flow_ind],
                    ):
                        flow = np.load(f)
                        valid = np.load(fm)
                        valid = 1 - valid / 100

                        cur_flows.append(flow)
                        cur_valids.append(valid)
                else:
                    for f in self.data[index].flows[flow_ind]:
                        flow, valid = frame_utils.readFlowKITTI(f)

                        cur_flows.append(flow)
                        cur_valids.append(valid)
            else:
                if self.dataset == "Infinigen":
                    # Inifinigen flow is stored as a 3D numpy array, [Flow, Depth]
                    for f in self.data[index].flows[flow_ind]:
                        flow = np.load(f)
                        flow = flow[..., :2]

                        cur_flows.append(flow)
                elif self.data[index].flow_mults[flow_ind] is not None:
                    for f, f_mult in zip(
                        self.data[index].flows[flow_ind],
                        self.data[index].flow_mults[flow_ind],
                    ):
                        flow = frame_utils.read_gen(f)
                        flow *= np.array(f_mult).astype(flow.dtype)
                        cur_flows.append(flow)
                else:
                    for f in self.data[index].flows[flow_ind]:
                        flow = frame_utils.read_gen(f)
                        cur_flows.append(flow)

            cur_flows = [np.array(flow).astype(np.float32) for flow in cur_flows]
            if not self.sparse:
                cur_valids = [
                    (
                        (np.abs(flow[:, :, 0]) < 1000) & (np.abs(flow[:, :, 1]) < 1000)
                    ).astype(np.float32)
                    for flow in cur_flows
                ]

            if self.subsample_groundtruth:
                # use only every second value in both spatial directions ==> flow will have same dimensions as images
                # used for spring dataset

                cur_flows = [flow[::2, ::2] for flow in cur_flows]
                cur_valids = [valid[::2, ::2] for valid in cur_valids]

            # TODO: merge flows
            if len(cur_flows) == 1:
                flows.append(cur_flows[0])
                valids.append(cur_valids[0])
            else:
                cur_flow = cur_flows[0]
                cur_valid = cur_valids[0]

                for i in range(1, len(cur_flows)):
                    cur_flow, cur_valid = merge_flows(
                        cur_flow, cur_valid, cur_flows[i], cur_valids[i]
                    )

                cur_flow = fill_invalid(cur_flow, cur_valid)

                flows.append(cur_flow)
                valids.append(cur_valid)

        # grayscale images
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]

        if self.augmentor is not None:
            if self.sparse:
                imgs, flows, valids = self.augmentor(imgs, flows, valids)
            else:
                imgs, flows, valids = self.augmentor(imgs, flows, valids)

        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]

        new_flows = []
        for flow in flows:
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            flow[torch.isnan(flow)] = 0
            flow[flow.abs() > 1e9] = 0
            new_flows.append(flow)
        flows = new_flows

        if len(valids):
            valids = [torch.from_numpy(valid) for valid in valids]
            if not self.sparse:
                valids = [
                    (valids[i] >= 0.5)
                    & (flows[i][0].abs() < 1000)
                    & (flows[i][1].abs() < 1000)
                    for i in range(len(flows))
                ]
        else:  # should never execute?
            valids = [(flow[0].abs() < 1000) & (flow[1].abs() < 1000) for flow in flows]

        return torch.stack(imgs), torch.stack(flows), torch.stack(valids).float()

    def add_datasets(self, other):
        self.data = self.data + other.data
        del other
        return self

    def __rmul__(self, v):
        self.data = v * self.data
        return self

    def __len__(self):
        return len(self.data)


class MpiSintel(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        split="train",
        root="datasets/Sintel",
        dstype="clean",
        scene_params={},
    ):
        super(MpiSintel, self).__init__(
            aug_params=aug_params, scene_params=scene_params
        )
        assert split in ["train", "val", "submission"]
        assert dstype in ["clean", "final"]
        self.dataset = "MpiSintel"

        seq_root = {
            "train": "training",
            "val": "training",
            "submission": "test",
        }[split]

        flow_root = osp.join(root, seq_root, "flow")
        image_root = osp.join(root, seq_root, dstype)

        if split == "submission":
            self.is_test = True

        scene_split = js_read(os.path.join("config", "splits", "sintel.json"))

        for scene in sorted(os.listdir(image_root)):
            if scene_split[scene] != split:
                continue

            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            flow_list = sorted(glob(osp.join(flow_root, scene, "*.flo")))

            current_scene = SceneData()
            current_scene.name = scene

            for i in range(len(image_list)):
                current_scene.images[(i, "left")] = image_list[i]

            if split != "submission":
                for i in range(len(flow_list)):
                    current_scene.flows[(i, "left")] = {(i + 1, "left"): flow_list[i]}

            self.scenes.append(current_scene)

        self.process_scenes(**scene_params)


class FlyingChairs(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        split="train",
        root="datasets/FlyingChairs/FlyingChairs_release/data",
        scene_params={},
    ):
        super(FlyingChairs, self).__init__(
            aug_params=aug_params, scene_params=scene_params
        )
        self.dataset = "FlyingChairs"

        images = sorted(glob(osp.join(root, "*.ppm")))
        flows = sorted(glob(osp.join(root, "*.flo")))
        assert len(images) // 2 == len(flows)

        split_list = np.loadtxt("chairs_split.txt", dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (
                split == "validation" and xid == 2
            ):
                current_scene = SceneData()
                current_scene.name = str(i)

                current_scene.images[(0, "left")] = images[2 * i]
                current_scene.images[(1, "left")] = images[2 * i + 1]
                current_scene.flows[(0, "left")] = {(1, "left"): flows[i]}

                self.scenes.append(current_scene)

        self.process_scenes(**scene_params)


class FlyingThings3D(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        root="datasets/FlyingThings3D",
        dstype="frames_cleanpass",
        scene_params={},
    ):
        super(FlyingThings3D, self).__init__(
            aug_params=aug_params, scene_params=scene_params
        )
        self.dataset = "FlyingThings3D"

        image_dirs, flow_dirs, disp_dirs = {}, {}, {}
        for cam in ["left", "right"]:
            image_dirs[cam] = sorted(glob(osp.join(root, dstype, "TRAIN/*/*")))
            image_dirs[cam] = sorted([osp.join(f, cam) for f in image_dirs[cam]])

            flow_dirs[cam] = {}
            for direction in ["into_future", "into_past"]:
                flow_dirs[cam][direction] = sorted(
                    glob(osp.join(root, "optical_flow/TRAIN/*/*"))
                )
                flow_dirs[cam][direction] = sorted(
                    [osp.join(f, direction, cam) for f in flow_dirs[cam][direction]]
                )

            disp_dirs[cam] = sorted(glob(osp.join(root, "disparity/TRAIN/*/*")))
            disp_dirs[cam] = sorted([osp.join(f, cam) for f in disp_dirs[cam]])

        for (
            idir_l,
            idir_r,
            fdir_fw_l,
            fdir_fw_r,
            fdir_bw_l,
            fdir_bw_r,
            ddir_l,
            ddir_r,
        ) in zip(
            image_dirs["left"],
            image_dirs["right"],
            flow_dirs["left"]["into_future"],
            flow_dirs["right"]["into_future"],
            flow_dirs["left"]["into_past"],
            flow_dirs["right"]["into_past"],
            disp_dirs["left"],
            disp_dirs["right"],
        ):
            images_l = sorted(glob(osp.join(idir_l, "*.png")))
            fw_flows_l = sorted(glob(osp.join(fdir_fw_l, "*.pfm")))
            bw_flows_l = sorted(glob(osp.join(fdir_bw_l, "*.pfm")))
            disp_l = sorted(glob(osp.join(ddir_l, "*.pfm")))

            images_r = sorted(glob(osp.join(idir_r, "*.png")))
            fw_flows_r = sorted(glob(osp.join(fdir_fw_r, "*.pfm")))
            bw_flows_r = sorted(glob(osp.join(fdir_bw_r, "*.pfm")))
            disp_r = sorted(glob(osp.join(ddir_r, "*.pfm")))

            current_scene = SceneData()

            for i in range(len(images_l)):
                current_scene.images[(i, "left")] = images_l[i]
                current_scene.flows[(i, "left")] = {
                    (i - 1, "left"): bw_flows_l[i],
                    (i + 1, "left"): fw_flows_l[i],
                }

                current_scene.images[(i, "right")] = images_r[i]
                current_scene.flows[(i, "right")] = {
                    (i - 1, "right"): bw_flows_r[i],
                    (i + 1, "right"): fw_flows_r[i],
                }

                current_scene.flows[(i, "left")][(i, "right")] = disp_l[i]
                current_scene.flows[(i, "right")][(i, "left")] = disp_r[i]

            for k in current_scene.flows:
                current_scene.flow_mults[k] = {}
                for k2 in current_scene.flows[k]:
                    current_scene.flow_mults[k][k2] = 1

            for i in range(len(images_l)):
                current_scene.flow_mults[(i, "right")][(i, "left")] = -1

            self.scenes.append(current_scene)

        self.process_scenes(**scene_params)


class KITTI(FlowDataset):
    def __init__(
        self, aug_params=None, split="train", root="datasets/KITTI", scene_params={}
    ):
        super(KITTI, self).__init__(
            aug_params=aug_params, scene_params=scene_params, sparse=True
        )
        assert split in ["train", "val", "test", "submission"]
        self.dataset = "KITTI"

        if split == "submission":
            self.is_test = True

        seq_split = {
            "train": "training",
            "val": "training",
            "test": "training",
            "submission": "testing",
        }[split]

        root = osp.join(root, seq_split)
        images0 = sorted(glob(osp.join(root, "image_2/*_09.png")))
        images1 = sorted(glob(osp.join(root, "image_2/*_10.png")))
        images2 = sorted(glob(osp.join(root, "image_2/*_11.png")))

        scene_split = js_read(os.path.join("config", "splits", "kitti.json"))

        image_list = []
        flow_list = []
        for img0, img1, img2 in zip(images0, images1, images2):
            if split != "submission" and scene_split[img1[-13:-7]] != split:
                continue

            image_list += [[img0, img1, img2]]

        if split != "submission":
            flow_list = [
                f
                for f in sorted(glob(osp.join(root, "flow_occ/*_10.png")))
                if scene_split[f[-13:-7]] == split
            ]

        for i in range(len(image_list)):
            current_scene = SceneData()
            current_scene.images[(0, "left")] = image_list[i][0]
            current_scene.images[(1, "left")] = image_list[i][1]
            current_scene.images[(2, "left")] = image_list[i][2]

            if split != "submission":
                current_scene.flows[(1, "left")] = {(2, "left"): flow_list[i]}

            self.scenes.append(current_scene)

        self.process_scenes(**scene_params)


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root="datasets/HD1K", scene_params={}):
        super(HD1K, self).__init__(
            aug_params=aug_params, scene_params=scene_params, sparse=True
        )
        self.dataset = "HD1K"

        seq_ix = 0
        while 1:
            flows = sorted(
                glob(os.path.join(root, "hd1k_flow_gt", "flow_occ/%06d_*.png" % seq_ix))
            )
            images = sorted(
                glob(os.path.join(root, "hd1k_input", "image_2/%06d_*.png" % seq_ix))
            )

            if len(flows) == 0:
                break

            current_scene = SceneData()

            for i in range(len(images)):
                current_scene.images[(i, "left")] = images[i]

            for i in range(len(flows)):
                current_scene.flows[(i, "left")] = {(i + 1, "left"): flows[i]}

            self.scenes.append(current_scene)
            seq_ix += 1

        self.process_scenes(**scene_params)


class SpringFlowDataset(FlowDataset):
    """
    Dataset class for Spring optical flow dataset.
    For train, this dataset returns image1, image2, flow and a data tuple (framenum, scene name, left/right cam, FW/BW direction).
    For test, this dataset returns image1, image2 and a data tuple (framenum, scene name, left/right cam, FW/BW direction).

    root: root directory of the spring dataset (should contain test/train directories)
    split: train/test split
    subsample_groundtruth: If true, return ground truth such that it has the same dimensions as the images (1920x1080px); if false return full 4K resolution
    """

    def __init__(
        self,
        aug_params=None,
        root="datasets/Spring",
        split="train",
        subsample_groundtruth=True,
        scene_params={},
    ):
        super(SpringFlowDataset, self).__init__(
            aug_params=aug_params, scene_params=scene_params
        )
        assert split in ["train", "val", "test", "submission"]

        self.dataset = "Spring"

        seq_root = {
            "train": "train",
            "val": "train",
            "test": "train",
            "submission": "test",
        }[split]

        seq_root = os.path.join(root, seq_root)

        if not os.path.exists(seq_root):
            raise ValueError(f"Spring directory does not exist: {seq_root}")

        self.subsample_groundtruth = subsample_groundtruth
        self.split = split
        self.seq_root = seq_root
        self.data_list = []
        if split == "submission":
            self.is_test = True

        scene_split = js_read(os.path.join("config", "splits", "spring.json"))

        for scene in sorted(os.listdir(seq_root)):
            if scene_split[scene] != split:
                continue

            current_scene = SceneData()
            current_scene.name = scene

            for cam in ["left", "right"]:
                images = sorted(
                    glob(os.path.join(seq_root, scene, f"frame_{cam}", "*.png"))
                )

                for i in range(len(images)):
                    current_scene.images[(i, cam)] = images[i]
                    current_scene.flows[(i, cam)] = {}
                    current_scene.flow_mults[(i, cam)] = {}

                if split != "submission":
                    for direction in ["FW"]:
                        flows = sorted(
                            glob(
                                os.path.join(
                                    seq_root, scene, f"flow_{direction}_{cam}", "*.flo5"
                                )
                            )
                        )

                        for i in range(len(flows)):
                            current_scene.flows[(i, cam)][(i + 1, cam)] = flows[i]
                            current_scene.flow_mults[(i, cam)][(i + 1, cam)] = 1

                    for direction in ["BW"]:
                        flows = sorted(
                            glob(
                                os.path.join(
                                    seq_root, scene, f"flow_{direction}_{cam}", "*.flo5"
                                )
                            )
                        )

                        for i in range(len(flows)):
                            current_scene.flows[(i + 1, cam)][(i, cam)] = flows[i]
                            current_scene.flow_mults[(i + 1, cam)][(i, cam)] = 1

                    if cam == "left":
                        othercam = "right"
                    else:
                        othercam = "left"

                    disps = sorted(
                        glob(os.path.join(seq_root, scene, f"disp1_{cam}", "*.dsp5"))
                    )

                    for i in range(len(disps)):
                        current_scene.flows[(i, cam)][(i, othercam)] = disps[i]
                        current_scene.flow_mults[(i, cam)][(i, othercam)] = (
                            1 if cam == "left" else -1
                        )

            self.scenes.append(current_scene)

        self.process_scenes(**scene_params)


class Infinigen(FlowDataset):
    def __init__(self, aug_params=None, root="datasets/Infinigen", scene_params={}):
        super(Infinigen, self).__init__(
            aug_params=aug_params, scene_params=scene_params
        )
        self.root = root
        scenes = glob(osp.join(self.root, "*/"))
        self.dataset = "Infinigen"
        for scene in sorted(scenes):
            if not osp.isdir(osp.join(scene, "frames")):
                continue

            current_scene = SceneData()

            images_left = sorted(glob(osp.join(scene, "frames/Image/camera_0/*.png")))
            images_right = sorted(glob(osp.join(scene, "frames/Image/camera_1/*.png")))
            for idx in range(len(images_left)):
                current_scene.images[(idx, "left")] = images_left[idx]
                current_scene.images[(idx, "right")] = images_right[idx]

            for idx in range(len(images_left) - 1):
                # name = Image + "_{ID}"
                ID_left = images_left[idx].split("/")[-1][6:-4]
                ID_right = images_right[idx].split("/")[-1][6:-4]
                flow_path_left = osp.join(
                    scene, "frames/Flow3D/camera_0", f"Flow3D_{ID_left}.npy"
                )
                flow_path_right = osp.join(
                    scene, "frames/Flow3D/camera_1", f"Flow3D_{ID_right}.npy"
                )
                current_scene.flows[(idx, "left")] = {(idx + 1, "left"): flow_path_left}
                current_scene.flows[(idx, "right")] = {
                    (idx + 1, "right"): flow_path_right
                }

            self.scenes.append(current_scene)

        self.process_scenes(**scene_params)


class TartanAir(FlowDataset):
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, aug_params=None, root="datasets/TartanAir", scene_params={}):
        super(TartanAir, self).__init__(
            aug_params=aug_params, scene_params=scene_params, sparse=True
        )
        self.dataset = "TartanAir"
        self.root = root
        self._build_dataset()

        self.process_scenes(**scene_params)

    def _build_dataset(self):
        scenes = glob(osp.join(self.root, "*/*/*"))

        for scene in sorted(scenes):
            current_scene = SceneData()
            current_scene.name = scene

            images = sorted(glob(osp.join(scene, "image_left/*.png")))
            for idx in range(len(images)):
                current_scene.images[(idx, "left")] = images[idx]

            for idx in range(len(images) - 1):
                frame0 = str(idx).zfill(6)
                frame1 = str(idx + 1).zfill(6)

                current_scene.flows[(idx, "left")] = {
                    (idx + 1, "left"): osp.join(
                        scene, "flow", f"{frame0}_{frame1}_flow.npy"
                    )
                }
                current_scene.flow_masks[(idx, "left")] = {
                    (idx + 1, "left"): osp.join(
                        scene, "flow", f"{frame0}_{frame1}_mask.npy"
                    )
                }

            self.scenes.append(current_scene)


# TODO: Adapt for new base class
class MegaScene(data.Dataset):
    def __init__(self, root_dir, npz_path, min_overlap_score=0.4, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info["pair_infos"].copy()
        del self.scene_info["pair_infos"]
        self.pair_infos = [
            pair_info
            for pair_info in self.pair_infos
            if pair_info[1] > min_overlap_score
        ]

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]
        img_name0 = osp.join(self.root_dir, self.scene_info["image_paths"][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info["image_paths"][idx1])
        depth_name0 = osp.join(self.root_dir, self.scene_info["depth_paths"][idx0])
        depth_name1 = osp.join(self.root_dir, self.scene_info["depth_paths"][idx1])
        # read intrinsics of original size
        K_0 = self.scene_info["intrinsics"][idx0].copy().reshape(3, 3)
        K_1 = self.scene_info["intrinsics"][idx1].copy().reshape(3, 3)
        # read and compute relative poses
        T0 = self.scene_info["poses"][idx0]
        T1 = self.scene_info["poses"][idx1]
        data = {
            "image0": img_name0,
            "image1": img_name1,
            "depth0": depth_name0,
            "depth1": depth_name1,
            "T0": T0,
            "T1": T1,  # (4, 4)
            "K0": K_0,  # (3, 3)
            "K1": K_1,
        }
        return data


# TODO: Adapt for new base class
class MegaDepth(FlowDataset):
    def __init__(self, aug_params=None, root="datasets/megadepth"):
        super(MegaDepth, self).__init__(aug_params, sparse=True)
        self.n_frames = 2
        self.dataset = "MegaDepth"
        self.root = root
        self._build_dataset()

    def _build_dataset(self):
        dataset_path = osp.join(self.root, "train")
        index_folder = osp.join(self.root, "index/scene_info_0.1_0.7")
        index_path_list = glob(index_folder + "/*.npz")
        dataset_list = []
        for index_path in index_path_list:
            my_dataset = MegaScene(dataset_path, index_path, min_overlap_score=0.4)
            dataset_list.append(my_dataset)

        self.megascene = torch.utils.data.ConcatDataset(dataset_list)
        for i in range(len(self.megascene)):
            data = self.megascene[i]
            self.image_list.append([data["image0"], data["image1"]])
            self.extra_info.append(
                [
                    data["depth0"],
                    data["depth1"],
                    data["T0"],
                    data["T1"],
                    data["K0"],
                    data["K1"],
                ]
            )


# TODO: Adapt for new base class
class Middlebury(FlowDataset):
    def __init__(self, aug_params=None, root="datasets/middlebury"):
        super(Middlebury, self).__init__(aug_params)
        img_root = os.path.join(root, "images")
        flow_root = os.path.join(root, "flow")

        flows = []
        imgs = []
        info = []

        for scene in sorted(os.listdir(flow_root)):
            img0 = os.path.join(img_root, scene, "frame10.png")
            img1 = os.path.join(img_root, scene, "frame11.png")
            flow = os.path.join(flow_root, scene, "flow10.flo")
            imgs += [(img0, img1)]
            flows += [flow]
            info += [scene]

        self.image_list = imgs
        self.flow_list = flows
        self.extra_info = info


def three_frame_wrapper1(dataset_class, dataset_args, add_reversed=True):
    datasets = []
    for cam in ["left", "right"]:
        datasets.append(
            dataset_class(
                **dataset_args,
                scene_params={
                    "frames": [(-1, cam), (0, cam), (1, cam)],
                    "flows": [((0, cam), (-1, cam)), ((0, cam), (1, cam))],
                    "invalid_images": "skip",
                },
            )
        )

        if add_reversed:
            datasets.append(
                dataset_class(
                    **dataset_args,
                    scene_params={
                        "frames": [(1, cam), (0, cam), (-1, cam)],
                        "flows": [((0, cam), (1, cam)), ((0, cam), (-1, cam))],
                        "invalid_images": "skip",
                    },
                )
            )

    return reduce(lambda x, y: x.add_datasets(y), datasets)


def three_frame_wrapper2(
    dataset_class, dataset_args, add_reversed=True, invalid_images="skip"
):
    datasets = []
    for cam in ["left", "right"]:
        datasets.append(
            dataset_class(
                **dataset_args,
                scene_params={
                    "frames": [(-1, cam), (0, cam), (1, cam)],
                    "flows": [None, ((0, cam), (1, cam))],
                    "invalid_images": invalid_images,
                },
            )
        )
        if add_reversed:
            datasets.append(
                dataset_class(
                    **dataset_args,
                    scene_params={
                        "frames": [(1, cam), (0, cam), (-1, cam)],
                        "flows": [((0, cam), (1, cam)), None],
                        "invalid_images": invalid_images,
                    },
                )
            )

    return reduce(lambda x, y: x.add_datasets(y), datasets)


def three_frame_wrapper_val(dataset_class, dataset_args):
    datasets = []
    for cam in ["left", "right"]:
        datasets.append(
            dataset_class(
                **dataset_args,
                scene_params={
                    "frames": [(-1, cam), (0, cam), (1, cam)],
                    "flows": [((0, cam), (1, cam))],
                },
            )
        )
        datasets.append(
            dataset_class(
                **dataset_args,
                scene_params={
                    "frames": [(1, cam), (0, cam), (-1, cam)],
                    "flows": [((0, cam), (-1, cam))],
                },
            )
        )

    return reduce(lambda x, y: x.add_datasets(y), datasets)


def three_frame_wrapper_spring_submission(dataset_class, dataset_args):
    datasets = []
    for cam in ["left", "right"]:
        datasets.append(
            dataset_class(
                **dataset_args,
                scene_params={
                    "frames": [(-1, cam), (0, cam), (1, cam)],
                    "flows": [],
                    "sequence_bounds": (0, 0),
                },
            )
        )
        datasets.append(
            dataset_class(
                **dataset_args,
                scene_params={
                    "frames": [(1, cam), (0, cam), (-1, cam)],
                    "flows": [],
                    "sequence_bounds": (1, 1),
                },
            )
        )

    return reduce(lambda x, y: x.add_datasets(y), datasets)


def three_frame_wrapper_sintel_submission(dataset_class, dataset_args):
    datasets = []
    for cam in ["left", "right"]:
        datasets.append(
            dataset_class(
                **dataset_args,
                scene_params={
                    "frames": [(-1, cam), (0, cam), (1, cam)],
                    "flows": [],
                    "sequence_bounds": (0, 0),
                },
            )
        )

    return reduce(lambda x, y: x.add_datasets(y), datasets)


def three_frame_wrapper_kitti_submission(dataset_class, dataset_args):
    datasets = []
    for cam in ["left", "right"]:
        datasets.append(
            dataset_class(
                **dataset_args,
                scene_params={
                    "frames": [(-1, cam), (0, cam), (1, cam)],
                    "flows": [],
                    "sequence_bounds": (0, 0),
                    "invalid_images": "skip"
                },
            )
        )

    return reduce(lambda x, y: x.add_datasets(y), datasets)


def fetch_dataloader(args):
    """Create the data loader for the corresponding training set"""

    if args.dataset == "things":
        aug_params = {
            "crop_size": args.image_size,
            "pre_scale": args.scale,
            "min_scale": -0.4,
            "max_scale": +0.8,
            "do_flip": True,
            "do_rotate": False,
        }

        clean_dataset = three_frame_wrapper1(
            FlyingThings3D, {"aug_params": aug_params, "dstype": "frames_cleanpass"}
        )
        final_dataset = three_frame_wrapper1(
            FlyingThings3D, {"aug_params": aug_params, "dstype": "frames_finalpass"}
        )

        train_dataset = clean_dataset + final_dataset

    elif args.dataset == "kitti":
        aug_params = {
            "crop_size": args.image_size,
            "pre_scale": args.scale,
            "min_scale": -0.2,
            "max_scale": +0.4,
            "do_flip": False,
        }
        train_dataset = three_frame_wrapper2(
            KITTI, {"aug_params": aug_params, "split": "train"}
        )

    elif args.dataset == "kitti-full":
        kitti = reduce(
            lambda x, y: x.add_datasets(y),
            [
                three_frame_wrapper2(
                    KITTI,
                    {
                        "aug_params": {
                            "crop_size": args.image_size,
                            "pre_scale": args.scale,
                            "min_scale": -0.2,
                            "max_scale": +0.4,
                            "do_flip": True,
                        },
                        "split": current_split,
                    },
                )
                for current_split in ["train", "val", "test"]
            ],
        )

        train_dataset = kitti

    elif args.dataset == "spring":
        aug_params = {
            "crop_size": args.image_size,
            "pre_scale": args.scale,
            "min_scale": 0.0,
            "max_scale": +0.2,
            "do_flip": True,
            "do_rotate": False,
        }
        train_dataset = three_frame_wrapper1(
            SpringFlowDataset, {"aug_params": aug_params, "subsample_groundtruth": True}
        )

    elif args.dataset == "spring-full":
        aug_params = {
            "crop_size": args.image_size,
            "pre_scale": args.scale,
            "min_scale": 0.0,
            "max_scale": +0.2,
            "do_flip": True,
            "do_rotate": False,
        }
        train_dataset = reduce(
            lambda x, y: x.add_datasets(y),
            [
                three_frame_wrapper1(
                    SpringFlowDataset,
                    {
                        "split": cur_sp,
                        "aug_params": aug_params,
                        "subsample_groundtruth": True,
                    },
                )
                for cur_sp in ["train", "val", "test"]
            ],
        )

    elif args.dataset == "TartanAir":
        aug_params = {
            "crop_size": args.image_size,
            "pre_scale": args.scale,
            "min_scale": -0.2,
            "max_scale": +0.4,
            "do_flip": True,
            "do_rotate": False,
        }
        train_dataset = three_frame_wrapper2(TartanAir, {"aug_params": aug_params})

    elif args.dataset == "TSKH":
        aug_params = {
            "crop_size": args.image_size,
            "pre_scale": args.scale,
            "min_scale": -0.2,
            "max_scale": +0.6,
            "do_flip": True,
            "do_rotate": False,
        }

        things_clean = three_frame_wrapper1(
            FlyingThings3D, {"aug_params": aug_params, "dstype": "frames_cleanpass"}
        )
        things_final = three_frame_wrapper1(
            FlyingThings3D, {"aug_params": aug_params, "dstype": "frames_finalpass"}
        )
        things = things_clean + things_final

        sintel_clean = three_frame_wrapper2(
            MpiSintel, {"aug_params": aug_params, "split": "train", "dstype": "clean"}
        )
        sintel_final = three_frame_wrapper2(
            MpiSintel, {"aug_params": aug_params, "split": "train", "dstype": "final"}
        )

        kitti = three_frame_wrapper2(
            KITTI,
            {
                "aug_params": {
                    "crop_size": args.image_size,
                    "pre_scale": args.scale,
                    "min_scale": -0.3,
                    "max_scale": +0.5,
                    "do_flip": True,
                },
                "split": "train",
            },
        )
        hd1k = three_frame_wrapper2(
            HD1K,
            {
                "aug_params": {
                    "crop_size": args.image_size,
                    "pre_scale": args.scale,
                    "min_scale": -0.5,
                    "max_scale": +0.2,
                    "do_flip": True,
                }
            },
        )

        # After (left + right) * (frames_cleanpass + frames_finalpass)
        train_dataset = (
            100 * sintel_clean + 100 * sintel_final + 400 * kitti + 120 * hd1k + things
        )

    elif args.dataset == "TSKH-full":
        aug_params = {
            "crop_size": args.image_size,
            "pre_scale": args.scale,
            "min_scale": -0.2,
            "max_scale": +0.6,
            "do_flip": True,
            "do_rotate": False,
        }

        things_clean = three_frame_wrapper1(
            FlyingThings3D, {"aug_params": aug_params, "dstype": "frames_cleanpass"}
        )
        things_final = three_frame_wrapper1(
            FlyingThings3D, {"aug_params": aug_params, "dstype": "frames_finalpass"}
        )
        things = things_clean + things_final

        sintel_clean = reduce(
            lambda x, y: x.add_datasets(y),
            [
                three_frame_wrapper2(
                    MpiSintel,
                    {
                        "aug_params": aug_params,
                        "split": current_split,
                        "dstype": "clean",
                    },
                )
                for current_split in ["train", "val"]
            ],
        )
        sintel_final = reduce(
            lambda x, y: x.add_datasets(y),
            [
                three_frame_wrapper2(
                    MpiSintel,
                    {
                        "aug_params": aug_params,
                        "split": current_split,
                        "dstype": "final",
                    },
                )
                for current_split in ["train", "val"]
            ],
        )

        kitti = reduce(
            lambda x, y: x.add_datasets(y),
            [
                three_frame_wrapper2(
                    KITTI,
                    {
                        "aug_params": {
                            "crop_size": args.image_size,
                            "pre_scale": args.scale,
                            "min_scale": -0.3,
                            "max_scale": +0.5,
                            "do_flip": True,
                        },
                        "split": current_split,
                    },
                )
                for current_split in ["train", "val", "test"]
            ],
        )

        hd1k = three_frame_wrapper2(
            HD1K,
            {
                "aug_params": {
                    "crop_size": args.image_size,
                    "pre_scale": args.scale,
                    "min_scale": -0.5,
                    "max_scale": +0.2,
                    "do_flip": True,
                }
            },
        )

        # After (left + right) * (frames_cleanpass + frames_finalpass)
        train_dataset = (
            80 * sintel_clean + 80 * sintel_final + 320 * kitti + 120 * hd1k + things
        )

    elif args.dataset == "sintel-full":
        aug_params = {
            "crop_size": args.image_size,
            "pre_scale": args.scale,
            "min_scale": -0.2,
            "max_scale": +0.6,
            "do_flip": True,
            "do_rotate": False,
        }

        sintel_clean = reduce(
            lambda x, y: x.add_datasets(y),
            [
                three_frame_wrapper2(
                    MpiSintel,
                    {
                        "aug_params": aug_params,
                        "split": current_split,
                        "dstype": "clean",
                    }
                )
                for current_split in ["train", "val"]
            ],
        )
        sintel_final = reduce(
            lambda x, y: x.add_datasets(y),
            [
                three_frame_wrapper2(
                    MpiSintel,
                    {
                        "aug_params": aug_params,
                        "split": current_split,
                        "dstype": "final",
                    }
                )
                for current_split in ["train", "val"]
            ],
        )

        train_dataset = sintel_clean + sintel_final
    else:
        raise ValueError(f"Invalid dataset name {args.dataset}")

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    print("Training with %d image pairs" % len(train_dataset))
    return train_loader
