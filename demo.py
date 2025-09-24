import subprocess
import time

import gradio as gr
import spaces
import torch
import tempfile
import cv2
import requests
from tqdm import tqdm

from memfof.utils.flow_viz import flow_to_image
from memfof.model import MEMFOF, AVAILABLE_MODELS


class FFmpegWriter:
    def __init__(self, output_path: str, width: int, height: int, fps: float):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None

    def __enter__(self):
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            self.output_path
        ]

        self.process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return self

    def write_frame(self, frame):
        """Write a single RGB24 frame to ffmpeg."""
        self.process.stdin.write(frame.tobytes())

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.process.stdin.close()
        except Exception as e:
            print(f"[ffmpeg] Failed to close stdin: {e}")
        finally:
            self.process.wait()


@torch.inference_mode()
def process_video(
    model: MEMFOF,
    input_path: str,
    output_path: str,
    device: torch.device,
    progress: gr.Progress | None = None,
    soft_duration: float = float("+inf")
):
    start_time = time.time()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    fmap_cache = [None] * 3

    pbar = tqdm(range(total_frames - 1), total=total_frames - 1)
    if progress is not None:
        pbar = progress.tqdm(pbar)

    with FFmpegWriter(output_path, width, height, fps) as writer:
        first_frame = True
        for _ in pbar:
            if time.time() - start_time >= soft_duration:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = torch.tensor(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                dtype=torch.float32
            ).permute(2, 0, 1).unsqueeze(0)

            if first_frame:
                frames.append(frame)
                first_frame = False

            frames.append(frame)

            if len(frames) != 3:
                continue

            frames_tensor = torch.stack(frames, dim=1).to(device)
            output = model(frames_tensor, fmap_cache=fmap_cache)

            forward_flow = output["flow"][-1][:, 1]  # FW [1, 2, H, W]
            flow_vis = flow_to_image(
                forward_flow.squeeze(dim=0).permute(1, 2, 0).cpu().numpy(),
                rad_min=0.02 * (height ** 2 + width ** 2) ** 0.5,
            )
            writer.write_frame(flow_vis)

            fmap_cache = output["fmap_cache"]
            fmap_cache.pop(0)
            fmap_cache.append(None)

            frames.pop(0)

    cap.release()


def download(url: str) -> str:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(tmp.name, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return tmp.name


@spaces.GPU(duration=60)
def run_demo(input_path: str, model_name: str) -> str:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = MEMFOF.from_pretrained(f"egorchistov/optical-flow-{model_name}").eval().to(device)
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    process_video(model, input_path, output_path, device, progress=gr.Progress(), soft_duration=57)
    return output_path


def main():
    videos = "https://msu-video-group.github.io/memfof/static/videos"
    davis_input = download(f"{videos}/davis_input.mp4")
    kitti_input = download(f"{videos}/kitti_input.mp4")
    sintel_input = download(f"{videos}/sintel_input.mp4")
    spring_input = download(f"{videos}/spring_input.mp4")

    video_input = gr.Video(
        label="Upload a video",
        value=davis_input,
    )
    checkpoint_dropdown = gr.Dropdown(
        label="Select checkpoint",
        choices=AVAILABLE_MODELS,
        value="MEMFOF-Tartan-T-TSKH"
    )
    video_output = gr.Video(label="Optical Flow")

    with gr.Blocks() as demo:
        gr.Markdown("""
        <h1 align="center">MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation</h1>

        <h2 align="center"><a href="https://arxiv.org/abs/2506.23151" style="text-decoration: none;">📄 Paper</a> | <a href="https://msu-video-group.github.io/memfof" style="text-decoration: none;">🌐 Project Page</a> | <a href="https://github.com/msu-video-group/memfof" style="text-decoration: none;">💻 Code</a> | <a href="https://colab.research.google.com/github/msu-video-group/memfof/blob/dev/demo.ipynb" style="text-decoration: none;">🚀 Colab</a></h2>

        <p align="center">Estimate optical flow using <b>MEMFOF</b> — a <b>memory-efficient optical flow model</b> for <b>Full HD video</b> that combines <b>high accuracy</b> with <b>low VRAM usage</b>.</p>
        
        <p align="center">Please note that the <b>processing will be automatically stopped after ~1 minute</b>.</p>
        """)

        with gr.Row():
            with gr.Column():
                video_input.render()
                checkpoint_dropdown.render()
                generate_btn = gr.Button("Estimate Optical Flow")
            video_output.render()

        generate_btn.click(
            fn=run_demo,
            inputs=[video_input, checkpoint_dropdown],
            outputs=video_output
        )

        gr.Examples(
            examples=[
                [kitti_input, "MEMFOF-Tartan-T-TSKH-kitti"],
                [sintel_input, "MEMFOF-Tartan-T-TSKH-sintel"],
                [spring_input, "MEMFOF-Tartan-T-TSKH-spring"],
            ],
            inputs=[video_input, checkpoint_dropdown],
            outputs=[video_output],
            fn=run_demo,
            cache_examples=True,
            cache_mode="lazy",
        )

    demo.launch()


if __name__ == "__main__":
    import os
    from huggingface_hub import login
    if "ACCESS_TOKEN" in os.environ:
        login(token=os.getenv("ACCESS_TOKEN"))
    main()
