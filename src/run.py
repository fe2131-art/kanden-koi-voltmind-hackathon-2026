import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from safety_agent.agent import AgentState, OpenAICompatLLM, build_agent
from safety_agent.modality_nodes import AudioAnalyzer, VisionAnalyzer, YOLODetector
from safety_agent.perceiver import Perceiver
from safety_agent.schema import CameraPose, Observation, ObservationProvider, WorldModel
from util.logger import setup_logger
from util.serializers import serialize_pydantic_or_dict

# Setup logger
logger = setup_logger("safety_view_agent")

# ==========================================
# Note: Video/audio constants are now loaded from configs/default.yaml
# ==========================================


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML config: {e}, using defaults")
        return {}
    except (IOError, OSError) as e:
        logger.warning(f"Failed to read config file: {e}, using defaults")
        return {}


def get_llm(config: dict) -> Optional[OpenAICompatLLM]:
    """Initialize LLM based on configuration and environment variables."""
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "openai")

    if provider == "openai":
        # OpenAI API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, using heuristic fallback")
            return None

        openai_cfg = llm_config.get("openai", {})
        model = os.getenv("OPENAI_MODEL", openai_cfg.get("model", "gpt-4o"))
        base_url = openai_cfg.get("base_url", "https://api.openai.com/v1")

        logger.info(f"Using OpenAI API (model={model})")
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
            logger.warning("LLM_BASE_URL not set, using heuristic fallback")
            return None

        api_key = vllm_cfg.get("api_key", "EMPTY")
        logger.info(f"Using vLLM server at {base_url} (model={model})")
        return OpenAICompatLLM(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=vllm_cfg.get("timeout_s", 60.0),
        )

    else:
        logger.warning(f"Unknown LLM provider: {provider}")
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
            logger.warning("OPENAI_API_KEY not set, VLM disabled")
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

        logger.info(f"Using VisionAnalyzer (model={model})")
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
            logger.warning("LLM_BASE_URL not set, VLM disabled")
            return None

        api_key = vlm_vllm.get("api_key") or llm_vllm.get("api_key", "EMPTY")
        timeout_s = vlm_vllm.get("timeout_s") or llm_vllm.get("timeout_s", 60.0)
        logger.info(f"Using VisionAnalyzer with vLLM at {base_url} (model={model})")
        return VisionAnalyzer(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    else:
        logger.warning(f"Unknown VLM provider: {provider}")
        return None


def find_video(search_dirs: list[str], video_extensions: set[str]) -> Optional[Path]:
    """Find video file in search directories.

    Args:
        search_dirs: List of directory paths to search
        video_extensions: Set of valid video file extensions (e.g., {".mp4", ".avi"})

    Returns:
        Path to first video file found, or None
    """
    for d in search_dirs:
        dir_path = Path(d)
        try:
            for f in sorted(dir_path.iterdir()):
                if f.is_file() and f.suffix.lower() in video_extensions:
                    logger.info(f"Found video: {f.name}")
                    return f
        except (FileNotFoundError, OSError, PermissionError):
            continue
    return None


def split_video_to_frames(
    video_path: str,
    frames_dir: str,
    frame_output_format: str = "frame_{timestamp}s.jpg",
    target_fps: float = 1.0,
    max_frames: int = 0,
    clear_frames: bool = False,
) -> tuple[list[Path], list[float]]:
    """Extract frames from video at specified FPS.

    Args:
        video_path: Path to video file
        frames_dir: Output directory for frames
        frame_output_format: Frame filename format template (e.g., "frame_{timestamp}s.jpg")
        target_fps: Target frames per second (e.g., 1.0 = 1 frame/sec)
        max_frames: Maximum frames to extract (0 = unlimited)
        clear_frames: If True, delete existing frames before extracting

    Returns:
        Tuple of (list of frame paths, list of video timestamps in seconds)
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing frames if requested
    if clear_frames:
        for f in frames_dir.glob("frame_*.jpg"):
            try:
                f.unlink()
            except OSError:
                pass

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return [], []

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if target_fps <= 0 or source_fps <= 0:
        logger.warning(f"Invalid FPS: source={source_fps}, target={target_fps}")
        cap.release()
        return [], []

    frame_interval = max(1, int(round(source_fps / target_fps)))

    frame_paths = []
    video_timestamps = []
    frame_count = 0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at specified interval
        if idx % frame_interval == 0:
            if max_frames > 0 and frame_count >= max_frames:
                break

            # Calculate timestamp in seconds
            timestamp = idx / source_fps
            timestamp_str = f"{timestamp:.3f}".rstrip('0').rstrip('.')

            # Save frame
            frame_filename = frame_output_format.format(timestamp=timestamp_str)
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            frame_paths.append(frame_path)
            video_timestamps.append(timestamp)
            frame_count += 1

        idx += 1

    cap.release()

    logger.info(f"Extracted {len(frame_paths)} frames from {Path(video_path).name} at {target_fps} FPS")
    return frame_paths, video_timestamps


def extract_audio(
    video_path: str,
    audio_dir: str,
    audio_output_filename: str = "audio.wav",
    audio_codec: str = "pcm_s16le",
    audio_sample_rate: int = 16000,
    audio_channels: int = 1,
) -> Optional[Path]:
    """Extract audio from video using ffmpeg.

    Args:
        video_path: Path to video file
        audio_dir: Output directory for audio
        audio_output_filename: Output audio filename
        audio_codec: Audio codec (e.g., "pcm_s16le")
        audio_sample_rate: Audio sample rate in Hz (e.g., 16000)
        audio_channels: Number of audio channels (e.g., 1 for mono)

    Returns:
        Path to extracted audio file, or None if no audio or ffmpeg not available
    """
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Check if ffprobe is available and if video has audio
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=5
        )
        if not result.stdout.strip():
            logger.warning(f"Video has no audio track: {Path(video_path).name}")
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("ffprobe not available (ffmpeg not installed)")
        return None

    # Extract audio using ffmpeg
    audio_path = audio_dir / audio_output_filename
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vn", "-acodec", audio_codec, "-ar", str(audio_sample_rate),
             "-ac", str(audio_channels), str(audio_path)],
            capture_output=True, timeout=30, check=True
        )
        logger.info(f"Extracted audio to {audio_path.name}")
        return audio_path
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Could not extract audio: {e}")
        return None


def load_frames(frames_dir: str = "data/frames") -> list[Path]:
    """Load frame images from frames directory.

    Args:
        frames_dir: Directory path to search for frame images

    Returns:
        List of Path objects to frame image files found, sorted
    """
    frames_path = Path(frames_dir)
    if not frames_path.exists():
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_files = sorted([
        f
        for f in frames_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if frame_files:
        logger.info(f"Found {len(frame_files)} frame(s) in {frames_dir}/")

    return frame_files


def save_analysis_results(
    output_dir: str,
    analysis_results: dict,
    video_timestamps: Optional[dict[str, float]] = None,
    agent_output: Optional[dict] = None,
) -> None:
    """Save analysis results to output directory (append mode for history).

    Args:
        output_dir: Output directory path
        analysis_results: Dict with 'perception_results' key containing list of results
        video_timestamps: Optional dict mapping obs_id to video timestamp in seconds
        agent_output: Optional dict with agent execution results (selected, world, plan, messages)
    """
    os.makedirs(output_dir, exist_ok=True)

    results_file = Path(output_dir) / "perception_results.json"
    default_data = {"perception_results": [], "agent_execution": []}

    # 既存データを読み込む（追記式）
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_data = default_data
    else:
        existing_data = default_data

    # Initialize agent_execution if it doesn't exist
    if "agent_execution" not in existing_data:
        existing_data["agent_execution"] = []

    # タイムスタンプを付与して新しいフレームを追加
    current_timestamp = time.time()
    for result in analysis_results.get("perception_results", []):
        result["timestamp"] = current_timestamp  # Unix timestamp（秒単位）

        # Add video_timestamp if available
        if video_timestamps:
            obs_id = result.get("obs_id")
            if obs_id and obs_id in video_timestamps:
                result["video_timestamp"] = video_timestamps[obs_id]

        existing_data["perception_results"].append(result)

    # Agent 実行結果を保存（提供されている場合）
    if agent_output:
        # Serialize messages (handle both dict and LangChain message objects)
        messages_list = []
        for msg in agent_output.get("messages", []):
            if isinstance(msg, dict):
                messages_list.append(msg)
            elif hasattr(msg, "content"):
                # LangChain message object
                messages_list.append({
                    "role": msg.__class__.__name__,
                    "content": getattr(msg, "content", str(msg))
                })
            else:
                messages_list.append({"content": str(msg)})

        agent_record = {
            "timestamp": current_timestamp,
            "step": agent_output.get("step", 0),
            "selected_view": None,
            "world_model": None,
            "plan": None,
            "messages": messages_list,
            "errors": agent_output.get("errors", []),
        }

        # Serialize agent output objects (using unified serializer)
        agent_record["selected_view"] = serialize_pydantic_or_dict(agent_output.get("selected"))
        agent_record["world_model"] = serialize_pydantic_or_dict(agent_output.get("world"))
        agent_record["plan"] = serialize_pydantic_or_dict(agent_output.get("plan"))

        existing_data["agent_execution"].append(agent_record)

    # ファイルに保存
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    frame_count = len(analysis_results.get('perception_results', []))
    agent_count = 1 if agent_output else 0
    logger.info(f"Results appended to {results_file} ({frame_count} frames, {agent_count} agent execution)")


def prepare_observations(
    config: dict,
    video_extensions: set,
    frame_output_format: str,
    audio_config: dict,
) -> tuple[list[Observation], dict]:
    """Prepare observation data from video and frames.

    Args:
        config: Configuration dictionary
        video_extensions: Set of valid video extensions
        frame_output_format: Frame filename format
        audio_config: Audio configuration

    Returns:
        Tuple of (observation list, video_timestamps_map)
    """
    video_cfg = config.get("video", {})
    audio_output_filename = audio_config.get("output_filename", "audio.wav")
    audio_codec = audio_config.get("codec", "pcm_s16le")
    audio_sample_rate = audio_config.get("sample_rate", 16000)
    audio_channels = audio_config.get("channels", 1)

    video_timestamps_map = {}

    # Process video
    video_path = find_video(["data/videos", "data"], video_extensions)
    if video_path:
        extract_audio(
            str(video_path),
            "data/audio",
            audio_output_filename=audio_output_filename,
            audio_codec=audio_codec,
            audio_sample_rate=audio_sample_rate,
            audio_channels=audio_channels,
        )
        frame_paths, video_timestamps = split_video_to_frames(
            str(video_path),
            "data/frames",
            frame_output_format=frame_output_format,
            target_fps=video_cfg.get("fps", 1.0),
            max_frames=video_cfg.get("max_frames", 30),
            clear_frames=video_cfg.get("clear_frames", False),
        )
        if frame_paths:
            video_timestamps_map = {f"img_{i}": ts for i, ts in enumerate(video_timestamps)}

    # Load frame images from data/frames
    frame_files = load_frames("data/frames")
    if frame_files:
        logger.info(f"Processing {len(frame_files)} frame image(s)")
        obs_list = [
            Observation(
                obs_id=f"img_{i}",
                image_path=str(frame_path.absolute()),
                audio_text=None,
                camera_pose=CameraPose(pan_deg=0, tilt_deg=0, zoom=1),
            )
            for i, frame_path in enumerate(frame_files)
        ]
        return obs_list, video_timestamps_map
    else:
        # Use example observations if no frames found (test fallback)
        logger.warning("No frames found in data/frames, using example observations")
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
        return obs_list, video_timestamps_map


def run_and_log_agent(
    agent: object,
    initial_state: AgentState,
    context: dict,
) -> tuple[dict, list[dict]]:
    """Run agent with streaming to collect all frame outputs.

    Args:
        agent: LangGraph agent instance
        initial_state: Initial agent state
        context: Agent runtime context

    Returns:
        Tuple of (final_state, all_frame_outputs)
    """
    logger.info("Running Safety View Agent")

    all_frame_outputs: list[dict] = []
    final_state: dict = {}
    prev_latest_obs_id: str | None = None

    for state in agent.stream(initial_state, context=context, stream_mode="values"):
        latest = state.get("latest_output")
        if "latest_output" in state and latest:
            obs_id = latest.get("obs_id")
            # フレームが更新されたときのみ追加（重複回避）
            if obs_id != prev_latest_obs_id:
                all_frame_outputs.append(latest)
                prev_latest_obs_id = obs_id
        final_state = state

    # Log agent results
    if final_state.get("selected"):
        logger.info(
            f"Selected view: {final_state['selected'].view_id} "
            f"(pan={final_state['selected'].pan_deg}°, tilt={final_state['selected'].tilt_deg}°)"
        )

    hazard_count = len(final_state["world"].fused_hazards) if final_state.get("world") and final_state["world"].fused_hazards else 0
    logger.info(f"Hazards detected: {hazard_count}")

    unobs_count = len(final_state["world"].outstanding_unobserved) if final_state.get("world") and final_state["world"].outstanding_unobserved else 0
    logger.info(f"Outstanding unobserved regions: {unobs_count}")

    if final_state.get("errors"):
        logger.warning(f"Errors: {len(final_state['errors'])}")

    return final_state, all_frame_outputs


def main():
    """Main entry point for Safety View Agent."""
    # Load configuration
    config = load_config()
    agent_cfg = config.get("agent", {})
    thresholds_cfg = config.get("thresholds", {})
    tokens_cfg = config.get("tokens", {})
    video_cfg = config.get("video", {})
    audio_cfg = config.get("audio", {})
    view_planning_cfg = config.get("view_planning", {})

    # Setup data directory
    os.makedirs("data", exist_ok=True)

    # Load video/audio formats from config
    video_formats_cfg = video_cfg.get("formats", {})
    video_extensions = set(video_formats_cfg.get("extensions", [".mp4", ".avi", ".mov", ".mkv", ".webm"]))
    frame_output_format = video_formats_cfg.get("frame_output", "frame_{timestamp}s.jpg")

    # Prepare observations from video and frames
    obs_list, video_timestamps_map = prepare_observations(
        config, video_extensions, frame_output_format, audio_cfg
    )

    # Apply max_steps filter to obs_list
    max_steps_cfg = agent_cfg.get("max_steps", 1)
    if max_steps_cfg == -1:
        actual_max_steps = len(obs_list)
    else:
        # N フレームだけ実行（obs_list を先頭から max_steps_cfg 件に制限）
        obs_list = obs_list[:max_steps_cfg]
        actual_max_steps = len(obs_list)

    if actual_max_steps > 0:
        logger.info(f"Configured to process {actual_max_steps} observation(s)")
    else:
        logger.warning("No observations to process")

    # Initialize agent components
    provider = ObservationProvider(obs_list)
    perceiver = Perceiver()
    llm = get_llm(config)
    vision_analyzer = get_vlm(config)

    # Build agent
    agent = build_agent()

    # Initialize modality analyzers for fan-out nodes
    audio_analyzer = AudioAnalyzer()
    yolo_detector = None
    if agent_cfg.get("enable_yolo", False):
        try:
            yolo_detector = YOLODetector("yolov8n.pt")
        except Exception as e:
            logger.warning(f"Failed to initialize YOLO: {e}, using fallback")

    # Initial state (with modality_results for fan-in)
    initial_state: AgentState = {
        "messages": [],
        "step": 0,
        "max_steps": actual_max_steps,
        "observation": None,
        "ir": None,
        "modality_results": {},
        "received_modalities": [],
        "barrier_obs_id": None,
        "latest_output": None,
        "world": WorldModel(),
        "plan": None,
        "selected": None,
        "done": False,
        "errors": [],
    }

    # Context (with modality analyzers for fan-out nodes)
    context = {
        "provider": provider,
        "perceiver": perceiver,
        "llm": llm,
        "vision_analyzer": vision_analyzer,
        "yolo_detector": yolo_detector,
        "audio_analyzer": audio_analyzer,
        "risk_stop_threshold": thresholds_cfg.get("risk_stop_threshold", 0.2),
        "hazard_focus_threshold": thresholds_cfg.get("hazard_focus_threshold", 0.6),
        "chat_max_tokens": tokens_cfg.get("chat_max_tokens", 2000),
        "max_outstanding_regions": view_planning_cfg.get("max_outstanding_regions", 6),
        "safety_priority_weight": view_planning_cfg.get("safety_priority_weight", 0.7),
        "info_gain_weight": view_planning_cfg.get("info_gain_weight", 0.3),
        "safety_priority_base": view_planning_cfg.get("safety_priority_base", 0.7),
        "expected_modalities": ["yolo", "vlm", "audio"],  # yolo/vlm に分割
        "run_mode": "until_provider_ends",  # provider が None を返すまで継続
    }

    # Run and log agent
    out, all_frame_outputs = run_and_log_agent(agent, initial_state, context)

    # Save all frame results from agent's emit_output node
    save_analysis_results(
        "data",
        {"perception_results": all_frame_outputs},
        video_timestamps_map,
        agent_output=out,
    )

    # Save Mermaid diagram
    try:
        mermaid_text = agent.get_graph().draw_mermaid()
        with open("data/flow.md", "w", encoding="utf-8") as f:
            f.write(f"# Safety View Agent Flow\n\n```mermaid\n{mermaid_text}\n```\n")
        logger.info("Graph diagram saved to data/flow.md")
    except Exception as e:
        logger.warning(f"Could not save graph diagram: {e}")


if __name__ == "__main__":
    main()
