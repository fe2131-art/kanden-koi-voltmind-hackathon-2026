"""Gradio-based frame-by-frame viewer for anomaly_samples.

Usage:
    python src/vis/anomaly_viewer.py
    python src/vis/anomaly_viewer.py --port 7861 --share

Opens a browser UI that lets you:
  - Select an anomaly class from a dropdown
  - Step through frames with Prev / Next buttons or a slider
  - See the class name, frame filename, and position (n / total)
"""

import argparse
from pathlib import Path

import gradio as gr

SAMPLES_DIR = Path("/home/team-005/data/hazard-detection/dataset/anomaly_samples")


def load_class_frames(class_name: str) -> list[Path]:
    """Return sorted list of image paths for the given class."""
    return sorted((SAMPLES_DIR / class_name).glob("*.jpg"))


def get_classes() -> list[str]:
    return sorted(d.name for d in SAMPLES_DIR.iterdir() if d.is_dir())


# ── Core render function ───────────────────────────────────────────────────

def render(class_name: str, idx: int) -> tuple:
    """Return (image_path, counter_text, updated_slider)."""
    frames = load_class_frames(class_name)
    if not frames:
        return None, "No frames found.", gr.update(minimum=0, maximum=0, value=0)
    idx = max(0, min(idx, len(frames) - 1))
    img = str(frames[idx])
    counter = f"{idx + 1} / {len(frames)}  —  {frames[idx].name}"
    slider = gr.update(minimum=0, maximum=len(frames) - 1, value=idx)
    return img, counter, slider


def on_class_change(class_name: str):
    return render(class_name, 0)


def on_prev(class_name: str, idx: int):
    return render(class_name, idx - 1)


def on_next(class_name: str, idx: int):
    return render(class_name, idx + 1)


def on_slider(class_name: str, idx: int):
    return render(class_name, int(idx))


# ── Build UI ───────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    classes = get_classes()
    default_class = classes[0] if classes else ""
    default_frames = load_class_frames(default_class) if default_class else []

    with gr.Blocks(title="Anomaly Frame Viewer") as demo:
        gr.Markdown("## Hazards & Robots — Anomaly Frame Viewer")
        gr.Markdown(
            "Select an anomaly class and step through frames with the buttons or slider."
        )

        with gr.Row():
            class_dd = gr.Dropdown(
                choices=classes,
                value=default_class,
                label="Anomaly class",
                scale=2,
            )
            counter_txt = gr.Textbox(
                value=f"1 / {len(default_frames)}" if default_frames else "—",
                label="Frame",
                interactive=False,
                scale=3,
            )

        image = gr.Image(
            value=str(default_frames[0]) if default_frames else None,
            label="Frame",
            type="filepath",
            height=512,
        )

        slider = gr.Slider(
            minimum=0,
            maximum=max(len(default_frames) - 1, 0),
            step=1,
            value=0,
            label="Frame index",
        )

        with gr.Row():
            prev_btn = gr.Button("◀  Prev", variant="secondary", scale=1)
            next_btn = gr.Button("Next  ▶", variant="primary", scale=1)

        # internal state: current frame index
        idx_state = gr.State(0)

        # ── Event wiring ──────────────────────────────────────────────────
        outputs = [image, counter_txt, slider]

        class_dd.change(
            fn=on_class_change,
            inputs=[class_dd],
            outputs=outputs,
        ).then(fn=lambda: 0, outputs=[idx_state])

        prev_btn.click(
            fn=on_prev,
            inputs=[class_dd, idx_state],
            outputs=outputs,
        ).then(fn=lambda s: max(0, s - 1), inputs=[idx_state], outputs=[idx_state])

        next_btn.click(
            fn=on_next,
            inputs=[class_dd, idx_state],
            outputs=outputs,
        ).then(
            fn=lambda s, cls: min(s + 1, len(load_class_frames(cls)) - 1),
            inputs=[idx_state, class_dd],
            outputs=[idx_state],
        )

        slider.release(
            fn=on_slider,
            inputs=[class_dd, slider],
            outputs=outputs,
        ).then(fn=lambda s: int(s), inputs=[slider], outputs=[idx_state])

    return demo


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Anomaly frame viewer")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(SAMPLES_DIR)],
    )


if __name__ == "__main__":
    main()
