import json
import os
import time
from pathlib import Path
from typing import Optional

import yaml

from safety_agent.agent import AgentState, OpenAICompatLLM, build_agent
from safety_agent.perceiver import Perceiver, VisionAnalyzer
from safety_agent.schema import CameraPose, Observation, ObservationProvider, WorldModel


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration."""
    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found: {config_path}, using defaults")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_llm(config: dict) -> Optional[OpenAICompatLLM]:
    """Initialize LLM based on configuration and environment variables."""
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "openai")

    if provider == "openai":
        # OpenAI API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set, using heuristic fallback")
            return None

        openai_cfg = llm_config.get("openai", {})
        model = os.getenv("OPENAI_MODEL", openai_cfg.get("model", "gpt-4o"))
        base_url = openai_cfg.get("base_url", "https://api.openai.com/v1")

        print(f"✅ Using OpenAI API (model={model})")
        return OpenAICompatLLM(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=openai_cfg.get("timeout_s", 60.0),
        )

    elif provider == "vllm":
        # Local vLLM server
        vllm_cfg = llm_config.get("vllm", {})
        base_url = os.getenv("LLM_BASE_URL", vllm_cfg.get("base_url"))
        model = os.getenv("LLM_MODEL", vllm_cfg.get("model"))

        if not base_url:
            print("⚠️  LLM_BASE_URL not set, using heuristic fallback")
            return None

        api_key = vllm_cfg.get("api_key", "EMPTY")
        print(f"✅ Using vLLM server at {base_url} (model={model})")
        return OpenAICompatLLM(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=vllm_cfg.get("timeout_s", 60.0),
        )

    else:
        print(f"⚠️  Unknown LLM provider: {provider}")
        return None


def get_vlm(config: dict) -> Optional[VisionAnalyzer]:
    """Initialize VLM (Vision Language Model) based on configuration and environment variables."""
    vlm_config = config.get("vlm", {})
    llm_config = config.get("llm", {})
    provider = vlm_config.get("provider", "openai")

    if provider == "openai":
        # OpenAI API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set, VLM disabled")
            return None

        vlm_openai = vlm_config.get("openai", {})
        llm_openai = llm_config.get("openai", {})

        # モデル: VLM設定 > 環境変数 > LLM設定(fallback) > デフォルト
        model = os.getenv(
            "VLM_MODEL",
            vlm_openai.get("model") or llm_openai.get("model", "gpt-5-nano-2025-08-07"),
        )

        # ベースURL: VLM設定 > LLM設定
        base_url = vlm_openai.get("base_url") or llm_openai.get(
            "base_url", "https://api.openai.com/v1"
        )

        # タイムアウト: VLM設定 > LLM設定
        timeout_s = vlm_openai.get("timeout_s") or llm_openai.get("timeout_s", 60.0)

        print(f"✅ Using VisionAnalyzer (model={model})")
        return VisionAnalyzer(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    elif provider == "vllm":
        # Local vLLM server (image support may vary)
        vlm_vllm = vlm_config.get("vllm", {})
        llm_vllm = llm_config.get("vllm", {})
        base_url = os.getenv(
            "LLM_BASE_URL", vlm_vllm.get("base_url") or llm_vllm.get("base_url")
        )
        model = os.getenv("VLM_MODEL", vlm_vllm.get("model") or llm_vllm.get("model"))

        if not base_url:
            print("⚠️  LLM_BASE_URL not set, VLM disabled")
            return None

        api_key = vlm_vllm.get("api_key") or llm_vllm.get("api_key", "EMPTY")
        timeout_s = vlm_vllm.get("timeout_s") or llm_vllm.get("timeout_s", 60.0)
        print(f"✅ Using VisionAnalyzer with vLLM at {base_url} (model={model})")
        return VisionAnalyzer(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    else:
        print(f"⚠️  Unknown VLM provider: {provider}")
        return None


def load_images_from_input(input_dir: str = "input") -> list:
    """Load images from input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"⚠️  Input directory not found: {input_dir}")
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_files = [
        f
        for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if image_files:
        print(f"✅ Found {len(image_files)} image(s) in {input_dir}/")
        return sorted(image_files)
    else:
        print(f"⚠️  No images found in {input_dir}/")
        return []


def save_analysis_results(output_dir: str, analysis_results: dict):
    """Save analysis results to output directory (append mode for history)."""
    os.makedirs(output_dir, exist_ok=True)

    results_file = Path(output_dir) / "perception_results.json"

    # 既存データを読み込む（追記式）
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_data = {"perception_results": []}
    else:
        existing_data = {"perception_results": []}

    # タイムスタンプを付与して新しいフレームを追加
    current_timestamp = time.time()
    for result in analysis_results.get("perception_results", []):
        result["timestamp"] = current_timestamp  # Unix timestamp（秒単位）
        existing_data["perception_results"].append(result)

    # ファイルに保存
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Analysis results appended to {results_file} ({len(analysis_results['perception_results'])} frames)")


def main():
    """Main entry point for Safety View Agent."""
    # Load configuration
    config = load_config()
    agent_cfg = config.get("agent", {})
    thresholds_cfg = config.get("thresholds", {})
    tokens_cfg = config.get("tokens", {})

    # Load images from input directory
    image_files = load_images_from_input("input")

    # Build observations from input images or use example observations
    if image_files:
        print("\n=== Processing Input Images ===\n")
        obs_list = [
            Observation(
                obs_id=f"img_{i}",
                image_path=str(img_path.absolute()),
                audio_text=None,
                camera_pose=CameraPose(pan_deg=0, tilt_deg=0, zoom=1),
            )
            for i, img_path in enumerate(image_files)
        ]
        # Get VLM for vision analysis
        vlm = get_vlm(config)

        # Process images with Perceiver and save results
        perceiver = Perceiver(enable_yolo=agent_cfg.get("enable_yolo", True), vlm=vlm)
        analysis_results = {
            "input_images": [str(p) for p in image_files],
            "perception_results": [],
        }

        for obs in obs_list:
            print(f"\n🔍 Processing: {Path(obs.image_path).name}")
            ir = perceiver.run(obs)

            print(f"   - Objects detected: {len(ir.objects)}")
            for obj in ir.objects:
                print(f"     • {obj.label} ({obj.confidence:.2%})")

            print(f"   - Hazards identified: {len(ir.hazards)}")
            for haz in ir.hazards:
                print(f"     • {haz.hazard_type} ({haz.confidence:.2%})")

            print(f"   - Unobserved regions: {len(ir.unobserved)}")

            # Vision analysis result from Perceiver (if VLM is available)
            vision_text = ir.vision_description or ""
            if ir.vision_description:
                print(
                    f"   ✅ Vision Analysis Complete ({len(ir.vision_description)} chars)"
                )

            analysis_results["perception_results"].append(
                {
                    "obs_id": ir.obs_id,
                    "objects": [obj.model_dump() for obj in ir.objects],
                    "hazards": [h.model_dump() for h in ir.hazards],
                    "unobserved": [u.model_dump() for u in ir.unobserved],
                    "audio": [a.model_dump() for a in ir.audio],
                    "vision_analysis": vision_text,
                }
            )

        # Save results to output
        save_analysis_results("output", analysis_results)

        # Continue with agent processing
        provider = ObservationProvider(obs_list)
        perceiver = Perceiver(enable_yolo=agent_cfg.get("enable_yolo", True), vlm=vlm)
    else:
        # Use example observations if no images found
        print("⚠️  Using example observations (no images in input/)\n")
        obs_list = [
            Observation(
                obs_id="t0",
                image_path=None,
                audio_text="I hear a car approach from the right",
                camera_pose=CameraPose(pan_deg=0, tilt_deg=0, zoom=1),
            ),
            Observation(
                obs_id="t1",
                image_path=None,
                audio_text=None,
                camera_pose=CameraPose(pan_deg=30, tilt_deg=0, zoom=1),
            ),
        ]
        provider = ObservationProvider(obs_list)
        vlm = get_vlm(config)
        perceiver = Perceiver(enable_yolo=agent_cfg.get("enable_yolo", False), vlm=vlm)

    # Initialize LLM based on config & environment variables
    llm = get_llm(config)

    # Build agent
    agent = build_agent()

    # Initial state
    initial_state: AgentState = {
        "messages": [],
        "step": 0,
        "max_steps": agent_cfg.get("max_steps", 3),
        "observation": provider.next(),
        "ir": None,
        "world": WorldModel(),
        "plan": None,
        "selected": None,
        "done": False,
        "errors": [],
    }

    # Context
    context = {
        "provider": provider,
        "perceiver": perceiver,
        "llm": llm,
        "risk_stop_threshold": thresholds_cfg.get("risk_stop_threshold", 0.2),
        "hazard_focus_threshold": thresholds_cfg.get("hazard_focus_threshold", 0.6),
        "chat_max_tokens": tokens_cfg.get(
            "chat_max_tokens", 2000
        ),  # LLM text generation
    }

    # Run agent
    print("\n=== Running Safety View Agent ===\n")
    out = agent.invoke(initial_state, context=context)

    # Output results
    print("\n=== Selected view command ===")
    print(out["selected"])
    print("\n=== World model ===")
    print(out["world"])
    print("\n=== Messages ===")
    for msg in out["messages"]:
        # Handle both dict and LangChain message objects
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", str(msg))
        )
        print(f"  {content}")
    if out["errors"]:
        print("\n=== Errors ===")
        for err in out["errors"]:
            print(f"  - {err}")

    # Save detailed analysis to output/
    try:
        os.makedirs("output", exist_ok=True)

        # Save agent execution summary
        summary_file = Path("output") / "agent_execution_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("SAFETY VIEW AGENT - EXECUTION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write("SELECTED VIEW COMMAND:\n")
            f.write("-" * 40 + "\n")
            if out["selected"]:
                f.write(f"View ID: {out['selected'].view_id}\n")
                f.write(
                    f"Pan: {out['selected'].pan_deg}°, Tilt: {out['selected'].tilt_deg}°, Zoom: {out['selected'].zoom}x\n"
                )
                f.write(f"Rationale: {out['selected'].why}\n\n")

            f.write("DETECTED HAZARDS:\n")
            f.write("-" * 40 + "\n")
            if out["world"].fused_hazards:
                for h in out["world"].fused_hazards:
                    f.write(f"• {h.hazard_type} (confidence: {h.confidence:.2%})\n")
                    if h.evidence:
                        f.write(f"  Evidence: {h.evidence}\n")
            else:
                f.write("No hazards detected\n")
            f.write("\n")

            f.write("OUTSTANDING UNOBSERVED REGIONS:\n")
            f.write("-" * 40 + "\n")
            if out["world"].outstanding_unobserved:
                for r in out["world"].outstanding_unobserved:
                    f.write(f"• {r.region_id}: {r.description}\n")
                    f.write(f"  Risk Level: {r.risk:.1%}\n")
            else:
                f.write("No unobserved regions\n")
            f.write("\n")

            f.write("AGENT MESSAGES LOG:\n")
            f.write("-" * 40 + "\n")
            for i, msg in enumerate(out["messages"], 1):
                # Handle both dict and LangChain message objects
                content = (
                    msg.get("content")
                    if isinstance(msg, dict)
                    else getattr(msg, "content", str(msg))
                )
                f.write(f"{i}. {content}\n")

            if out["errors"]:
                f.write("\nERRORS:\n")
                f.write("-" * 40 + "\n")
                for err in out["errors"]:
                    f.write(f"⚠️  {err}\n")

        print("✅ Agent execution summary saved to output/agent_execution_summary.txt")

        # Save Mermaid diagram
        mermaid_text = agent.get_graph().draw_mermaid()
        with open("output/flow.md", "w", encoding="utf-8") as f:
            f.write(f"# Safety View Agent Flow\n\n```mermaid\n{mermaid_text}\n```\n")
        print("✅ Graph diagram saved to output/flow.md")

    except Exception as e:
        print(f"\n⚠️  Could not save outputs: {e}")


if __name__ == "__main__":
    main()
