import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import yaml
from dotenv import load_dotenv

from safety_agent.agent import AgentState, OpenAICompatLLM, build_agent
from safety_agent.modality_nodes import (
    AudioAnalyzer,
    DepthEstimator,
    VisionAnalyzer,
)
from safety_agent.schema import CameraPose, Observation, ObservationProvider
from tts.synthesize import synthesize_frame
from util.logger import setup_logger

# .env ファイルから環境変数を読み込む
load_dotenv()

logger = setup_logger("safety_view_agent", level=logging.DEBUG)

# ==========================================
# 注: ビデオ・音声定数は configs/default.yaml から読み込まれます
# ==========================================


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """YAML設定を読み込む。ファイル未検出・構文エラーは即 raise する。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"設定ファイルが見つかりません: {config_path}\n"
            "リポジトリルートから実行しているか確認してください。"
        )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            raise ValueError(f"設定ファイルが空です: {config_path}")
        return cfg
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 構文エラー ({config_path}): {e}") from e
    except (IOError, OSError) as e:
        raise IOError(f"設定ファイル読み込み失敗 ({config_path}): {e}") from e


def load_prompts(prompt_path: str = "configs/prompt.yaml") -> dict:
    """プロンプト設定を読み込む。ファイル未検出・構文エラーは即 raise する。"""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(
            f"プロンプト設定ファイルが見つかりません: {prompt_path}\n"
            "configs/prompt.yaml を作成してください。"
        )
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            raise ValueError(f"プロンプト設定ファイルが空です: {prompt_path}")
        return cfg
    except yaml.YAMLError as e:
        raise ValueError(f"プロンプト YAML 構文エラー ({prompt_path}): {e}") from e


def get_llm(config: dict) -> Optional[OpenAICompatLLM]:
    """設定と環境変数に基づいて LLM を初期化。"""
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "openai")

    if provider == "openai":
        # OpenAI API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, using default assessment")
            return None

        openai_cfg = llm_config.get("openai", {})
        model = openai_cfg.get("model")
        if not model:
            raise ValueError(
                "LLM モデル名が未設定です。"
                "configs/default.yaml の llm.openai.model を設定してください。"
            )
        base_url = openai_cfg.get("base_url")
        if not base_url:
            raise ValueError(
                "LLM ベースURLが未設定です。"
                "configs/default.yaml の llm.openai.base_url を設定してください。"
            )

        logger.info(f"Using OpenAI API (model={model})")
        return OpenAICompatLLM(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=openai_cfg.get("timeout_s", 60.0),
        )

    elif provider == "vllm":
        # Local vLLM server（Structured Outputs で JSON 安定化）
        vllm_cfg = llm_config.get("vllm", {})
        base_url = vllm_cfg.get("base_url")
        model = vllm_cfg.get("model")

        if not base_url:
            logger.warning(
                "LLM base_url not set in configs/default.yaml, using default assessment"
            )
            return None
        if not model:
            logger.warning(
                "LLM model not set in configs/default.yaml, using default assessment"
            )
            return None

        api_key = vllm_cfg.get("api_key", "EMPTY")
        logger.info(f"Using vLLM server at {base_url} (model={model})")
        return OpenAICompatLLM(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=vllm_cfg.get("timeout_s", 60.0),
            is_vllm=True,  # vLLM の Structured Outputs を有効化
        )

    else:
        logger.warning(f"Unknown LLM provider: {provider}")
        return None


def get_vlm(config: dict, prompts: dict) -> Optional[VisionAnalyzer]:
    """VLM（Vision Language Model）を設定と環境変数に基づいて初期化。"""
    vlm_config = config.get("vlm", {})
    llm_config = config.get("llm", {})
    provider = vlm_config.get("provider", "openai")

    # vision_analysis プロンプトを取得（必須）
    vision_prompt = prompts.get("vision_analysis", {}).get("default_prompt")
    if vision_prompt is None:
        raise ValueError(
            "プロンプト設定 vision_analysis.default_prompt が見つかりません。"
            "configs/prompt.yaml を確認してください。"
        )

    # tokens_cfg から vision_max_completion_tokens を取得（必須）
    tokens_cfg = config.get("tokens", {})
    vision_max_tokens = tokens_cfg.get("vision_max_completion_tokens")
    if vision_max_tokens is None:
        raise ValueError(
            "tokens.vision_max_completion_tokens が未設定です。"
            "configs/default.yaml を確認してください。"
        )

    if provider == "openai":
        # OpenAI API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, VLM disabled")
            return None

        vlm_openai = vlm_config.get("openai", {})
        llm_openai = llm_config.get("openai", {})

        # モデル: VLM設定 > 環境変数 > LLM設定(fallback)
        model = (
            os.getenv("VLM_MODEL") or vlm_openai.get("model") or llm_openai.get("model")
        )
        if not model:
            raise ValueError(
                "VLM モデル名が未設定です。"
                "configs/default.yaml の vlm.openai.model または環境変数 VLM_MODEL を設定してください。"
            )

        # ベースURL: VLM設定 > LLM設定
        _vlm_base = vlm_openai.get("base_url")
        _llm_base = llm_openai.get("base_url")
        base_url = _vlm_base if _vlm_base is not None else _llm_base
        if not base_url:
            raise ValueError(
                "VLM ベースURLが未設定です。"
                "configs/default.yaml の vlm.openai.base_url または llm.openai.base_url を設定してください。"
            )

        # タイムアウト: VLM設定 > LLM設定
        _vlm_timeout = vlm_openai.get("timeout_s")
        _llm_timeout = llm_openai.get("timeout_s")
        timeout_s = _vlm_timeout if _vlm_timeout is not None else _llm_timeout
        if timeout_s is None:
            raise ValueError(
                "VLM タイムアウトが未設定です。"
                "configs/default.yaml の vlm.openai.timeout_s を設定してください。"
            )

        logger.info(f"Using VisionAnalyzer (model={model})")
        return VisionAnalyzer(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout_s,
            default_prompt=vision_prompt.strip(),
            max_tokens=vision_max_tokens,
            provider="openai",
        )

    elif provider == "vllm":
        # Local vLLM server (image support may vary)
        vlm_vllm = vlm_config.get("vllm", {})
        llm_vllm = llm_config.get("vllm", {})

        # ベースURL: VLM設定 > 環境変数 > LLM設定
        _vlm_url = vlm_vllm.get("base_url")
        _llm_url = llm_vllm.get("base_url")
        base_url = _vlm_url if _vlm_url is not None else _llm_url

        # モデル: VLM設定 > 環境変数 > LLM設定
        model = vlm_vllm.get("model") or llm_vllm.get("model")

        if not base_url:
            logger.warning("LLM_BASE_URL not set, VLM disabled")
            return None

        # API キーとタイムアウト
        _vlm_api = vlm_vllm.get("api_key")
        _llm_api = llm_vllm.get("api_key")
        api_key = _vlm_api if _vlm_api is not None else (_llm_api or "EMPTY")

        _vlm_timeout = vlm_vllm.get("timeout_s")
        _llm_timeout = llm_vllm.get("timeout_s")
        timeout_s = _vlm_timeout if _vlm_timeout is not None else (_llm_timeout or 60.0)

        logger.info(f"Using VisionAnalyzer with vLLM at {base_url} (model={model})")
        return VisionAnalyzer(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout_s,
            default_prompt=vision_prompt.strip(),
            max_tokens=vision_max_tokens,
            provider="vllm",
        )

    else:
        logger.warning(f"Unknown VLM provider: {provider}")
        return None


def get_alm(config: dict, prompts: dict) -> Optional[AudioAnalyzer]:
    """ALM（Audio Language Model）を設定と環境変数に基づいて初期化。"""
    alm_config = config.get("alm", {})
    llm_config = config.get("llm", {})
    audio_config = config.get("audio", {})
    provider = alm_config.get("provider", "openai")

    # audio_analysis プロンプトを取得（必須）
    audio_prompt = prompts.get("audio_analysis", {}).get("default_prompt")
    if audio_prompt is None:
        raise ValueError(
            "プロンプト設定 audio_analysis.default_prompt が見つかりません。"
            "configs/prompt.yaml を確認してください。"
        )

    sample_rate = audio_config.get("sample_rate", 16000)
    window_seconds = audio_config.get("window_seconds", 3.0)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, ALM disabled")
            return None

        alm_openai = alm_config.get("openai", {})
        llm_openai = llm_config.get("openai", {})

        model = (
            os.getenv("ALM_MODEL") or alm_openai.get("model") or llm_openai.get("model")
        )
        if not model:
            raise ValueError(
                "ALM モデル名が未設定です。"
                "configs/default.yaml の alm.openai.model または環境変数 ALM_MODEL を設定してください。"
            )

        _alm_base = alm_openai.get("base_url")
        _llm_base = llm_openai.get("base_url")
        base_url = _alm_base if _alm_base is not None else _llm_base
        if not base_url:
            raise ValueError(
                "ALM ベースURLが未設定です。"
                "configs/default.yaml の alm.openai.base_url または llm.openai.base_url を設定してください。"
            )

        _alm_timeout = alm_openai.get("timeout_s")
        _llm_timeout = llm_openai.get("timeout_s")
        timeout_s = _alm_timeout if _alm_timeout is not None else _llm_timeout
        if timeout_s is None:
            raise ValueError(
                "ALM タイムアウトが未設定です。"
                "configs/default.yaml の alm.openai.timeout_s を設定してください。"
            )

        logger.info(f"Using AudioAnalyzer (model={model})")
        return AudioAnalyzer(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout_s,
            sample_rate=sample_rate,
            window_seconds=window_seconds,
            default_prompt=audio_prompt.strip(),
            provider="openai",
        )

    elif provider == "vllm":
        alm_vllm = alm_config.get("vllm", {})
        llm_vllm = llm_config.get("vllm", {})

        _alm_url = alm_vllm.get("base_url")
        _llm_url = llm_vllm.get("base_url")
        base_url = _alm_url if _alm_url is not None else _llm_url

        model = alm_vllm.get("model") or llm_vllm.get("model")

        if not base_url:
            logger.warning("ALM base_url not set, ALM disabled")
            return None

        _alm_api = alm_vllm.get("api_key")
        _llm_api = llm_vllm.get("api_key")
        api_key = _alm_api if _alm_api is not None else (_llm_api or "EMPTY")

        _alm_timeout = alm_vllm.get("timeout_s")
        _llm_timeout = llm_vllm.get("timeout_s")
        timeout_s = _alm_timeout if _alm_timeout is not None else (_llm_timeout or 60.0)

        logger.info(f"Using AudioAnalyzer with vLLM at {base_url} (model={model})")
        return AudioAnalyzer(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout_s,
            sample_rate=sample_rate,
            window_seconds=window_seconds,
            default_prompt=audio_prompt.strip(),
            provider="vllm",
        )

    else:
        logger.warning(f"Unknown ALM provider: {provider}")
        return None


def find_video(search_dirs: list[str], video_extensions: set[str]) -> Optional[Path]:
    """検索ディレクトリからビデオファイルを探す。

    Args:
        search_dirs: 検索対象ディレクトリパスのリスト
        video_extensions: 有効なビデオファイル拡張子のセット（例: {".mp4", ".avi"}）

    Returns:
        見つかった最初のビデオファイルのパス、または None
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
    """ビデオから指定 FPS でフレームを抽出。

    Args:
        video_path: ビデオファイルのパス
        frames_dir: フレーム出力ディレクトリ
        frame_output_format: フレームファイル名形式テンプレート（例: "frame_{timestamp}s.jpg"）
        target_fps: 目標フレームレート（例: 1.0 = 1フレーム/秒）
        max_frames: 抽出最大フレーム数（0 = 無制限）
        clear_frames: True の場合、抽出前に既存フレームを削除

    Returns:
        (フレームパスリスト, ビデオタイムスタンプリスト（秒単位）) のタプル
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
        cap.release()
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

            # Calculate timestamp in seconds (keep 1 decimal place = 0.1s unit)
            timestamp = idx / source_fps
            timestamp_str = f"{timestamp:.1f}"

            # Save frame
            frame_filename = frame_output_format.format(timestamp=timestamp_str)
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            frame_paths.append(frame_path)
            video_timestamps.append(timestamp)
            frame_count += 1

        idx += 1

    cap.release()

    logger.info(
        f"Extracted {len(frame_paths)} frames from {Path(video_path).name} at {target_fps} FPS"
    )
    return frame_paths, video_timestamps


def extract_audio(
    video_path: str,
    audio_dir: str,
    audio_output_filename: str = "audio.wav",
    audio_codec: str = "pcm_s16le",
    audio_sample_rate: int = 16000,
    audio_channels: int = 1,
) -> Optional[Path]:
    """ffmpeg を使用してビデオから音声を抽出。

    Args:
        video_path: ビデオファイルのパス
        audio_dir: 音声出力ディレクトリ
        audio_output_filename: 出力音声ファイル名
        audio_codec: 音声コーデック（例: "pcm_s16le"）
        audio_sample_rate: 音声サンプリングレート（Hz、例: 16000）
        audio_channels: 音声チャンネル数（例: 1 はモノラル）

    Returns:
        抽出した音声ファイルのパス、またはなし（音声がないか ffmpeg が利用不可の場合）
    """
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Check if ffprobe is available and if video has audio
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
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
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                audio_codec,
                "-ar",
                str(audio_sample_rate),
                "-ac",
                str(audio_channels),
                str(audio_path),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        logger.info(f"Extracted audio to {audio_path.name}")
        return audio_path
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:
        logger.warning(f"Could not extract audio: {e}")
        return None


def load_frames(frames_dir: str = "data/frames") -> list[Path]:
    """フレームディレクトリから画像を読み込む。

    Args:
        frames_dir: フレーム画像を検索するディレクトリパス

    Returns:
        見つかったフレーム画像ファイルのパスオブジェクトのリスト（ソート済み）
    """
    frames_path = Path(frames_dir)
    if not frames_path.exists():
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_files = sorted(
        [
            f
            for f in frames_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    )

    if frame_files:
        logger.info(f"Found {len(frame_files)} frame(s) in {frames_dir}/")

    return frame_files


def save_analysis_results(
    output_dir: str,
    analysis_results: dict,
    video_timestamps: Optional[dict[str, float]] = None,
    _agent_output: Optional[dict] = None,
) -> None:
    """分析結果を出力ディレクトリに保存（追記モードで履歴を保持）。

    Args:
        output_dir: 出力ディレクトリパス
        analysis_results: 結果リストを含む 'frames' キーの辞書
        video_timestamps: obs_id をビデオタイムスタンプ（秒単位）にマッピングする辞書（オプション）
        _agent_output: 使用されていません（互換性のため保持）
    """
    os.makedirs(output_dir, exist_ok=True)

    results_file = Path(output_dir) / "perception_results.json"
    default_data = {"frames": []}

    # 既存データを読み込む（追記式）
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_data = default_data
    else:
        existing_data = default_data

    # Ensure frames key exists (migration support)
    if "frames" not in existing_data:
        existing_data["frames"] = []

    # タイムスタンプを付与して新しいフレームを追加
    current_timestamp = time.time()
    for result in analysis_results.get("frames", []):
        result["timestamp"] = current_timestamp  # Unix timestamp（秒単位）

        # Add video_timestamp if available
        if video_timestamps:
            frame_id = result.get("frame_id")
            if frame_id and frame_id in video_timestamps:
                result["video_timestamp"] = video_timestamps[frame_id]

        existing_data["frames"].append(result)

    # ファイルに保存（アトミック書き込みで破損防止）
    try:
        tmp = results_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, results_file)
    except (IOError, TypeError, OSError) as e:
        logger.error(f"Failed to save results to {results_file}: {e}")
        return
    frame_count = len(analysis_results.get("frames", []))
    logger.info(f"Results appended to {results_file} ({frame_count} frames)")


def prepare_observations_inspesafe(
    config: dict,
    frame_output_format: str,
    audio_config: dict,
) -> tuple[list[Observation], dict]:
    """InspecSafe-V1 モード: セッションパスから動画・音声を取得して Observation を構築。

    Args:
        config: 設定辞書
        frame_output_format: フレームファイル名形式テンプレート
        audio_config: 音声設定

    Returns:
        (観測リスト、ビデオタイムスタンプマップ) のタプル
    """
    data_cfg = config.get("data", {})
    inspesafe_cfg = data_cfg.get("inspesafe", {})

    # 設定値の取得
    dataset_path = Path(inspesafe_cfg.get("dataset_path", "../InspecSafe-V1"))
    session_rel = inspesafe_cfg.get("session", "")
    if not session_rel:
        raise ValueError("data.inspesafe.session が設定されていません")

    # セッションディレクトリを解決（DATA_PATH/ を補完）
    session_dir = dataset_path / "DATA_PATH" / session_rel
    if not session_dir.exists():
        raise FileNotFoundError(f"セッションが見つかりません: {session_dir}")

    # 赤外線動画からフレーム抽出（利用可能な場合）
    # TODO: 赤外線フレーム統合機能は将来実装予定（マルチモーダル分析に統合）
    video_cfg = config.get("video", {})
    infrared_videos = sorted(session_dir.glob("*_infrared_*.mp4"))
    if infrared_videos:
        _process_infrared_inspesafe(infrared_videos[0], video_cfg, frame_output_format)

    # 動画ファイル（*_visible_*.mp4）を取得
    rgb_videos = sorted(session_dir.glob("*_visible_*.mp4"))
    if not rgb_videos:
        raise FileNotFoundError(f"RGB 動画が見つかりません: {session_dir}")
    video_path = rgb_videos[0]
    logger.info(f"[inspesafe] 動画: {video_path}")

    # 音声ファイル（*_audio_*.wav）を data/audio/ にコピー
    audio_files = sorted(session_dir.glob("*_audio_*.wav"))
    audio_dir = Path("data/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_output_filename = audio_config.get("output_filename", "audio.wav")
    audio_dest = audio_dir / audio_output_filename

    if audio_files:
        shutil.copy2(str(audio_files[0]), str(audio_dest))
        logger.info(f"[inspesafe] 音声コピー: {audio_files[0]} → {audio_dest}")
    else:
        logger.warning(f"[inspesafe] 音声ファイルなし: {session_dir}")

    # フレーム展開（既存の split_video_to_frames を再利用）
    frames, video_timestamps = split_video_to_frames(
        video_path=str(video_path),
        frames_dir="data/frames",
        frame_output_format=frame_output_format,
        target_fps=video_cfg.get("fps", 1.0),
        max_frames=video_cfg.get("max_frames", 30),
        clear_frames=video_cfg.get("clear_frames", False),
    )
    logger.info(f"[inspesafe] {len(frames)} フレーム展開完了")

    video_timestamps_map = {f"img_{i}": ts for i, ts in enumerate(video_timestamps)}

    # Observation リスト構築（load_frames を再利用）
    frame_paths = load_frames("data/frames")
    obs_list = [
        Observation(
            obs_id=f"img_{i}",
            image_path=str(fp.resolve()),
            prev_image_path=str(frame_paths[i - 1].resolve()) if i > 0 else None,
            audio_path="data/audio/audio.wav",
            camera_pose=CameraPose(pan_deg=0, tilt_deg=0, zoom=1),
            video_timestamp=video_timestamps_map.get(f"img_{i}"),
        )
        for i, fp in enumerate(frame_paths)
    ]
    return obs_list, video_timestamps_map


def _process_infrared_inspesafe(
    infrared_video_path: Path, video_cfg: dict, frame_output_format: str
) -> tuple[list[Path], dict]:
    """InspecSafe-V1 モード用：赤外線動画からフレーム抽出処理。

    Args:
        infrared_video_path: 赤外線動画ファイルパス
        video_cfg: ビデオ設定辞書
        frame_output_format: フレームファイル名形式テンプレート

    Returns:
        (フレームパスリスト, ビデオタイムスタンプマップ) のタプル
    """
    infrared_frames_dir = "data/infrared_frames"

    # Clean up existing infrared frames
    if os.path.exists(infrared_frames_dir):
        shutil.rmtree(infrared_frames_dir)
    os.makedirs(infrared_frames_dir, exist_ok=True)

    logger.info(f"[inspesafe] 赤外線動画: {infrared_video_path.name}")

    # Extract frames from infrared video (no audio extraction needed)
    frame_paths, video_timestamps = split_video_to_frames(
        str(infrared_video_path),
        infrared_frames_dir,
        frame_output_format=frame_output_format,
        target_fps=video_cfg.get("fps", 1.0),
        max_frames=video_cfg.get("max_frames", 30),
        clear_frames=False,  # Already cleaned above
    )

    if frame_paths:
        logger.info(f"[inspesafe] {len(frame_paths)} 赤外線フレーム展開完了")
        # Create timestamp map (same format as RGB processing)
        infrared_timestamps_map = {
            f"ir_{i}": ts for i, ts in enumerate(video_timestamps)
        }
        return frame_paths, infrared_timestamps_map
    else:
        logger.warning(
            f"No frames extracted from infrared video: {infrared_video_path}"
        )
        return [], {}


def prepare_observations(
    config: dict,
    video_extensions: set,
    frame_output_format: str,
    audio_config: dict,
) -> tuple[list[Observation], dict]:
    """ビデオとフレームから観測データを準備。

    Args:
        config: 設定辞書
        video_extensions: 有効なビデオ拡張子のセット
        frame_output_format: フレームファイル名形式
        audio_config: 音声設定

    Returns:
        (観測リスト、ビデオタイムスタンプマップ) のタプル
    """
    # データ入力モードの判定
    data_mode = config.get("data", {}).get("mode", "manual")

    if data_mode == "inspesafe":
        return prepare_observations_inspesafe(config, frame_output_format, audio_config)

    # 以降は manual モード処理
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
            video_timestamps_map = {
                f"img_{i}": ts for i, ts in enumerate(video_timestamps)
            }

    # Load frame images from data/frames
    frame_files = load_frames("data/frames")
    if frame_files:
        logger.info(f"Processing {len(frame_files)} frame image(s)")
        obs_list = [
            Observation(
                obs_id=f"img_{i}",
                image_path=str(frame_path.absolute()),
                prev_image_path=str(frame_files[i - 1].absolute()) if i > 0 else None,
                audio_path="data/audio/audio.wav",
                camera_pose=CameraPose(pan_deg=0, tilt_deg=0, zoom=1),
                video_timestamp=video_timestamps_map.get(f"img_{i}"),
            )
            for i, frame_path in enumerate(frame_files)
        ]
        return obs_list, video_timestamps_map
    else:
        raise FileNotFoundError(
            "フレームが見つかりません: data/frames/\n"
            "以下のいずれかを実施してください：\n"
            "1. data/videos/ にビデオファイルを配置し、split_video_to_frames() を実行\n"
            "2. data/frames/ に画像ファイルを直接配置\n"
            "3. configs/default.yaml で data.mode = 'inspesafe' に設定"
        )


def run_and_log_agent(
    agent: object,
    initial_state: AgentState,
    context: dict,
    on_frame_callback=None,
) -> tuple[dict, list[dict]]:
    """ストリーミングでエージェントを実行して全フレーム出力を収集。

    Args:
        agent: LangGraph エージェントインスタンス
        initial_state: 初期エージェント状態
        context: エージェント実行時コンテキスト
        on_frame_callback: フレーム処理後に呼び出される関数（frame_output を引数とする）

    Returns:
        (最終状態、全フレーム出力) のタプル
    """
    logger.info("Running Safety View Agent")

    all_frame_outputs: list[dict] = []
    final_state: dict = {}
    prev_latest_obs_id: str | None = None

    try:
        for state in agent.stream(initial_state, context=context, stream_mode="values"):
            latest = state.get("latest_output")
            if "latest_output" in state and latest:
                frame_id = latest.get("frame_id")
                # フレームが更新されたときのみ追加（重複回避）
                if frame_id != prev_latest_obs_id:
                    all_frame_outputs.append(latest)
                    prev_latest_obs_id = frame_id
                    # フレーム処理後、コールバックがあれば即時実行
                    if on_frame_callback:
                        on_frame_callback(latest)
            final_state = state
    except Exception as e:
        logger.error(f"Agent streaming error: {e}", exc_info=True)
        # 収集済みのフレームを返す

    # Log agent results
    if final_state.get("selected"):
        logger.info(
            f"Selected view: {final_state['selected'].view_id} "
            f"(pan={final_state['selected'].pan_deg}°, tilt={final_state['selected'].tilt_deg}°)"
        )

    if final_state.get("assessment"):
        logger.info(
            f"Assessment: {final_state['assessment'].action_type} risk={final_state['assessment'].risk_level}"
        )

    if final_state.get("errors"):
        logger.warning(f"Errors: {len(final_state['errors'])}")

    return final_state, all_frame_outputs


def main():
    """Safety View Agent のメインエントリーポイント。"""
    # Load configuration
    config = load_config()
    prompts = load_prompts()
    agent_cfg = config.get("agent", {})
    tokens_cfg = config.get("tokens", {})
    video_cfg = config.get("video", {})
    audio_cfg = config.get("audio", {})

    # Setup data directory
    os.makedirs("data", exist_ok=True)

    # Archive existing perception_results.json with timestamp before starting new run
    perception_results_file = "data/perception_results.json"
    results_archive_dir = "data/results_archive"
    if os.path.exists(perception_results_file):
        os.makedirs(results_archive_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archived_file = os.path.join(
            results_archive_dir, f"perception_results_{timestamp}.json"
        )
        shutil.move(perception_results_file, archived_file)
        logger.info(f"Archived perception_results.json → {archived_file}")

    # Clean up data directories before processing
    # Note: data/audio is NOT cleared as it contains source audio files used by multiple runs
    for data_dir in ["data/frames", "data/depth", "data/voice"]:
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    # Load video/audio formats from config
    video_formats_cfg = video_cfg.get("formats", {})
    video_extensions = set(
        video_formats_cfg.get("extensions", [".mp4", ".avi", ".mov", ".mkv", ".webm"])
    )
    frame_output_format = video_formats_cfg.get(
        "frame_output", "frame_{timestamp}s.jpg"
    )

    # Prepare observations from video and frames
    try:
        obs_list, video_timestamps_map = prepare_observations(
            config, video_extensions, frame_output_format, audio_cfg
        )
    except FileNotFoundError as e:
        logger.error(f"Data preparation failed: {e}")
        sys.exit(1)

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
    llm = get_llm(config)
    vision_analyzer = get_vlm(config, prompts)
    audio_analyzer = (
        get_alm(config, prompts) if agent_cfg.get("enable_audio", False) else None
    )

    # Build agent
    agent = build_agent()

    # Initialize modality analyzers for fan-out nodes
    depth_estimator = None
    if agent_cfg.get("enable_depth", False):
        try:
            depth_estimator = DepthEstimator()
        except Exception as e:
            logger.warning(
                f"Failed to initialize DepthEstimator: {e}, depth estimation will not be available"
            )

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
        "last_vision_summary": None,
        "last_assessment": None,
        "assessment": None,
        "done": False,
        "errors": [],
    }

    # Build expected_modalities based on enabled features
    expected_modalities = ["vlm"]  # Vision is always expected
    if agent_cfg.get("enable_audio", False):
        expected_modalities.append("audio")
    if agent_cfg.get("enable_depth", False):
        expected_modalities.append("depth")

    # Context (with modality analyzers for fan-out nodes)
    context = {
        "provider": provider,
        "llm": llm,
        "vision_analyzer": vision_analyzer,
        "audio_analyzer": audio_analyzer,
        "depth_estimator": depth_estimator,
        "prompts": prompts,
        "config": config,  # depth_node で config.get("tokens", ...) 使用
        "chat_max_tokens": tokens_cfg.get("chat_max_tokens", 2000),
        "context_history_size": agent_cfg.get("context_history_size", 1),
        "expected_modalities": expected_modalities,
        "run_mode": "until_provider_ends",  # provider が None を返すまで継続
    }

    # TTS 初期化（フレーム単位合成用）
    # サーバモード専用（ローカルモード対応は廃止）
    tts_cfg = config.get("tts", {})
    tts_server_url: Optional[str] = tts_cfg.get("server_url") or None

    # フレーム処理時のコールバック関数定義
    def _on_frame(frame_output: dict) -> None:
        """フレーム処理完了時に JSON 保存 + TTS 合成を即時実行"""
        save_analysis_results(
            "data",
            {"frames": [frame_output]},
            video_timestamps_map,
        )
        try:
            synthesize_frame(
                frame=frame_output,
                outdir=Path("data/voice"),
                model=None,
                server_url=tts_server_url,
                voice=tts_cfg.get("voice", "Vivian"),
                language=tts_cfg.get("language", "Japanese"),
                instruct=tts_cfg.get("instructions") or tts_cfg.get("instruct") or None,
                sample_rate=int(tts_cfg.get("sample_rate", 12000)),
                temperature=tts_cfg.get("temperature"),
                top_p=tts_cfg.get("top_p"),
                top_k=tts_cfg.get("top_k"),
                repetition_penalty=tts_cfg.get("repetition_penalty"),
                task_type=tts_cfg.get("task_type") or None,
            )
        except Exception as e:
            logger.error(
                f"フレーム {frame_output.get('frame_id')} の TTS 処理失敗: {e}"
            )

    # Run and log agent with per-frame callback
    _, all_frame_outputs = run_and_log_agent(
        agent,
        initial_state,
        context,
        on_frame_callback=_on_frame,
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
