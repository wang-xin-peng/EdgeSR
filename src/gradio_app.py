"""
Gradio web demo for EdgeSR.

Usage:
    python src/gradio_app.py --config configs/edgesr_standard.yaml --model edgesr --checkpoint checkpoints/edgesr_standard_best.pt
"""

import os
import argparse
import yaml
import torch
import gradio as gr
import numpy as np
from PIL import Image
import tempfile

from src.models import EDSRBaseline, EdgeSR, EdgeSRNoLCAP
from src.models.edgesr_pruned import prune_model


def get_model(config, checkpoint_path, device):
    model_name = config["model"]["name"]
    if model_name == "baseline":
        model = EDSRBaseline(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            scale=config["data"]["scale"],
        )
    elif model_name == "edgesr":
        model = EdgeSR(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            n_earb=config["model"]["n_earb"],
            scale=config["data"]["scale"],
            lcap_threshold=config["model"]["lcap_threshold"],
        )
    elif model_name == "edgesr_nolcap":
        model = EdgeSRNoLCAP(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            n_earb=config["model"]["n_earb"],
            scale=config["data"]["scale"],
        )
    elif model_name == "edgesr_pruned":
        base = EdgeSR(
            n_resblocks=config["model"]["n_resblocks"],
            n_feats=config["model"]["n_feats"],
            n_earb=config["model"]["n_earb"],
            scale=config["data"]["scale"],
            lcap_threshold=0.01,
        )
        model = prune_model(base, threshold=config["model"].get("prune_threshold", 0.5))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def process_image(model, input_img, scale_factor, device):
    """
    Process a single image through the SR model.

    Args:
        model: torch model
        input_img: PIL Image or numpy array
        scale_factor: 2 or 4
        device: torch device

    Returns:
        (input_display, output_pil): tuple of display images
    """
    if isinstance(input_img, np.ndarray):
        input_img = Image.fromarray(input_img)

    img_tensor = torch.from_numpy(np.array(input_img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    _, _, h, w = img_tensor.shape
    pad_h = (scale_factor - h % scale_factor) % scale_factor
    pad_w = (scale_factor - w % scale_factor) % scale_factor
    if pad_h > 0 or pad_w > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="replicate")
    img_tensor = img_tensor.to(device)
    sr_tensor = model(img_tensor)
    sr_tensor = sr_tensor.clamp(0, 1).cpu()
    sr_h = h * scale_factor
    sr_w = w * scale_factor
    sr_tensor = sr_tensor[:, :, :sr_h, :sr_w]
    sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).numpy()
    sr_pil = Image.fromarray((sr_np * 255).astype(np.uint8))
    input_display = input_img.resize(sr_pil.size, Image.BICUBIC)
    return input_display, sr_pil


def create_demo(model, device):
    """Create the Gradio interface."""
    def super_resolve(input_img, scale_factor):
        if input_img is None:
            return None, None, None
        try:
            scale_factor = int(scale_factor)
            input_display, sr_pil = process_image(model, input_img, scale_factor, device)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                sr_pil.save(tmp.name)
                sr_path = tmp.name
            return input_display, sr_pil, sr_path
        except Exception as e:
            raise gr.Error(f"Processing failed: {str(e)}")

    with gr.Blocks(title="EdgeSR - Image Super-Resolution", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # EdgeSR: 轻量级图像超分辨率系统
            **上传一张图片，选择放大倍数，一键生成高清图像。**
            """
        )
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="输入图像", type="pil")
                scale_selector = gr.Radio(
                    choices=[2, 4],
                    value=2,
                    label="放大倍数",
                    info="2x (推荐) 或 4x"
                )
                submit_btn = gr.Button("✨ 生成超分辨率图像", variant="primary")
            with gr.Column():
                with gr.Tab("对比视图"):
                    gallery = gr.Gallery(
                        label="对比 (左: 输入放大 | 右: 超分结果)",
                        columns=2,
                        height="auto",
                        object_fit="contain",
                    )
                with gr.Tab("结果"):
                    sr_output = gr.Image(label="超分辨率输出", type="pil")
                download_btn = gr.DownloadButton(label="📥 下载结果", variant="secondary")

        sr_path_state = gr.State()

        def process_and_show(input_img, scale_factor):
            input_display, sr_pil, sr_path = super_resolve(input_img, scale_factor)
            return [input_display, sr_pil], sr_pil, sr_path

        submit_btn.click(
            fn=process_and_show,
            inputs=[input_img, scale_selector],
            outputs=[gallery, sr_output, sr_path_state],
            show_progress='hidden',
        )
        download_btn.click(
            fn=lambda path: path,
            inputs=[sr_path_state],
            outputs=[download_btn],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/edgesr_standard.yaml")
    parser.add_argument("--model", type=str, default="edgesr", choices=["baseline", "edgesr", "edgesr_nolcap", "edgesr_pruned"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["model"]["name"] = args.model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model = get_model(config, args.checkpoint, device)
    print("Model loaded. Launching Gradio demo...")

    demo = create_demo(model, device)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
