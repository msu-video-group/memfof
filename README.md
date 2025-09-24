<p align="center">
  <h1 align="center">MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation</h1>
  <p align="center">
    <a href="https://github.com/VladBargin">Vladislav Bargatin</a>
    ·
    <a href="http://github.com/egorchistov">Egor Chistov</a>
    ·
    <a href="https://github.com/AlexanderYakovenko1">Alexander Yakovenko</a>
    ·
    <a href="https://linkedin.com/in/dmitriyvatolin">Dmitriy Vatolin</a>
  </p>
  <h3 align="center">ICCV 2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2506.23151">📄 Paper</a> | <a href="https://msu-video-group.github.io/memfof">🌐 Project Page</a> | <a href="https://colab.research.google.com/github/msu-video-group/memfof/blob/dev/demo.ipynb">🚀 Colab</a> | <a href="https://huggingface.co/spaces/egorchistov/optical-flow-MEMFOF">🤗 Demo</a> | <a href="https://huggingface.co/collections/egorchistov/optical-flow-memfof-685695802e71b207b96d8fb8">📦 Models</a></h3>
</p>

## 🏅 Overview

**MEMFOF** is a **memory-efficient optical flow method** for **Full HD video** that combines **high accuracy** with **low VRAM usage**.

## 🤗 Demo

Given a video sequence, our code can estimate its optical flow. Visit the [demo page](https://huggingface.co/spaces/egorchistov/MEMFOF) and try it with your own video.

> 🏞️ Prefer MEMFOF-Tartan-T-TSKH model for real-world videos — it is trained with higher diversity and robustness in mind.

## 🚀 Using MEMFOF in Your Project

Install MEMFOF via the package manager:

```shell
pip3 install git+https://github.com/msu-video-group/memfof
```

Then use the following snippet to compute backward and forward optical flow for three consecutive frames:

```python
import torch
from memfof import MEMFOF

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MEMFOF.from_pretrained("egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH").eval().to(device)

with torch.inference_mode():
    # [B=1, T=3, C=3, H=1080, W=1920]
    example_input = torch.randint(0, 256, [1, 3, 3, 1080, 1920], device=device)
    # [B=1, C=2, H=1080, W=1920]
    backward_flow, forward_flow = model(example_input)["flow"][-1].unbind(dim=1)
```

Refer to the [demo notebook](https://colab.research.google.com/github/msu-video-group/memfof/blob/dev/demo.ipynb) for quick start.

## 📦 Models

- [`MEMFOF-Tartan`](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan)
- [`MEMFOF-Tartan-T`](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T)
- [`MEMFOF-Tartan-T-TSKH`](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH) (✅ Best for real videos)
- [`MEMFOF-Tartan-T-TSKH-kitti`](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-kitti)
- [`MEMFOF-Tartan-T-TSKH-sintel`](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-sintel)
- [`MEMFOF-Tartan-T-TSKH-spring`](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-spring)

## 🛠️ Dev Installation

To train, evaluate, or submit MEMFOF, you’ll need the dev installation. Run the following commands:

```shell
git clone https://github.com/msu-video-group/memfof.git
cd memfof
pip3 install --editable .[dev]
```

## 🗂️ Datasets

To train MEMFOF, you will need to download the required datasets: [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [Sintel](http://sintel.is.tue.mpg.de/), [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow), [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/), [TartanAir](https://theairlab.org/tartanair-dataset/), and [Spring](https://spring-benchmark.org/).

By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder.

```shell
├── datasets
    ├── Sintel
    ├── KITTI
    ├── FlyingThings3D
    ├── HD1K
    ├── Spring
        ├── test
        ├── train
    ├── TartanAir
```

## 📊 Evaluation and Submission

Please refer to [eval.sh](eval.sh) and [submission.sh](submission.sh) for more details.

## 🏋️ Training

Our training setup is configured to use a fixed effective batch size with **4 nodes 8 GPUs each**.
You can train the model with fewer resources (no need to alter the configs), but if you encounter **out-of-memory (OOM)** errors, try increasing the `accumulate_grad_batches` parameter in the configs. For example, set it to 4 when training on a single node with 8 GPUs.

By default training is configured for submissions using the `*-full` versions of the benchmark datasets.
For experiments and ablations however, it is recommended to switch to the versions without the `-full` postfix to get validation results on a separate validation set.

Our training script is optimized for use with the slurm workload manager. A typical submission script looks like this:

```shell
# (submit.sh)
#!/bin/bash

#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16

srun bash train.sh
```

Alternatively, multi-node training is also supported via other launch methods, such as torchrun:

```shell
OMP_NUM_THREADS=16 torchrun \
--nproc_per_node=8 \
--nnodes=4 \
--node_rank <NODE_RANK> \
--master_addr <MASTER_ADDR> \
--master_port <MASTER_PORT> \
--no-python bash train.sh
```

For more details, refer to the [PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/2.5.1/clouds/cluster.html).

We use Weights & Biases (WandB) for experiment tracking by default. To disable logging, set the environment variable:

```shell
export WANDB_MODE=disabled
```

## ❓ Need Help?

Feel free to open an issue if you have any questions.

## 📚 Citation

```
@article{bargatin2025memfof,
  title={MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation},
  author={Bargatin, Vladislav and Chistov, Egor and Yakovenko, Alexander and Vatolin, Dmitriy},
  journal={arXiv preprint arXiv:2506.23151},
  year={2025}
}
```

## 🙏 Acknowledgements

This project relies on code from existing repositories: [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT), [VideoFlow](https://github.com/XiaoyuShi97/VideoFlow),  and [GMA](https://github.com/zacjiang/GMA). We thank the original authors for their excellent work.
