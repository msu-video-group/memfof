import argparse
import os
import cv2

import torch
from tqdm import tqdm

from config.parser import parse_args
from memfof import datasets
from memfof.model_lit import MEMFOFLit
from memfof.utils import frame_utils
from memfof.utils.flow_viz import flow_to_image


@torch.inference_mode()
def create_spring_submission(model: MEMFOFLit, device: str, output_path: str):
    """Create submission for the Spring leaderboard"""
    test_dataset = datasets.three_frame_wrapper_spring_submission(
        datasets.SpringFlowDataset, {"split": "submission"}
    )
    for test_id in tqdm(range(len(test_dataset))):
        images, extra_info = test_dataset[test_id]
        scene, frame, _, frames, _ = extra_info
        images = images.unsqueeze(0).to(device)

        flow, _ = model.scale_and_forward_flow(images, scale=0)

        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        flow_gt_vis = flow_to_image(flow, convert_to_bgr=True)

        cam = frames[0][1]
        if frames[0][0] < 0:
            direction = "FW"
        else:
            direction = "BW"

        output_dir = os.path.join(output_path, scene, f"flow_{direction}_{cam}")
        output_file = os.path.join(
            output_dir, f"flow_{direction}_{cam}_{frame + 1:04d}.flo5"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cv2.imwrite(
            os.path.join(output_dir, f"flow_{direction}_{cam}_{frame + 1:04d}.png"),
            flow_gt_vis,
        )
        frame_utils.writeFlo5File(flow, output_file)


@torch.inference_mode()
def create_sintel_submission(model: MEMFOFLit, device: str, output_path: str):
    """Create submission for the Sintel leaderboard"""
    for dstype in ["clean", "final"]:
        test_dataset = datasets.three_frame_wrapper_sintel_submission(
            datasets.MpiSintel, {"split": "submission", "dstype": dstype}
        )
        for test_id in tqdm(range(len(test_dataset))):
            images, extra_info = test_dataset[test_id]
            scene, frame, _, _, _ = extra_info
            images = images.unsqueeze(0).to(device)

            flow, _ = model.scale_and_forward_flow(images, scale=1)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            flow_gt_vis = flow_to_image(flow, convert_to_bgr=True)

            output_dir = os.path.join(output_path, dstype, scene)
            output_file = os.path.join(output_dir, "frame%04d.flo" % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            cv2.imwrite(os.path.join(output_dir, f"frame{frame + 1}.png"), flow_gt_vis)


@torch.inference_mode()
def create_kitti_submission(model: MEMFOFLit, device: str, output_path):
    """Create submission for the Sintel leaderboard"""
    test_dataset = datasets.three_frame_wrapper_kitti_submission(
        datasets.KITTI, {"split": "submission", "aug_params": None}
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in tqdm(range(len(test_dataset))):
        images, _ = test_dataset[test_id]
        frame = f"{test_id:06d}_10.png"
        images = images.unsqueeze(0).to(device)

        flow, _ = model.scale_and_forward_flow(images, scale=1)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        flow_gt_vis = flow_to_image(flow, convert_to_bgr=True)

        output_filename = os.path.join(output_path, frame)
        cv2.imwrite(os.path.join(output_path, f"frame{frame}"), flow_gt_vis)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.inference_mode()
def eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MEMFOFLit(args).to(device).eval()
    output_path = os.path.join(args.output_dir, args.dataset)

    if args.dataset == "spring":
        create_spring_submission(model, device, output_path)
    elif args.dataset == "sintel":
        create_sintel_submission(model, device, output_path)
    elif args.dataset == "kitti":
        create_kitti_submission(model, device, output_path)
    else:
        raise ValueError(f"Unkown dataset {args.dataset} requested for evaluation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str, help="Saving path for checkpoints", nargs="?", default="submissions")
    parser.add_argument("--cfg", help="experiment config file name", required=True, type=str)
    args = parser.parse_args()
    args = parse_args(parser)
    os.makedirs(args.output_dir, exist_ok=True)
    eval(args)


if __name__ == "__main__":
    main()
