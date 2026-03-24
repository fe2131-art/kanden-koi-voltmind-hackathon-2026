import os
import yaml
import shutil
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


def main(config_path: str = "audio_processing/audio_processing.yaml") -> None:
    """
    Copy original session directory and preprocess audio file with background and anomaly noise.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # copy original session directory
    original_config = config.get("original")
    original_dir = original_config.get("dir")
    session_names = original_config.get("session_names")
    print("Preprocessing following sessions.")
    print(session_names)
    print()

    for session_name in session_names:
        copy_session_name = session_name + "_audio"
        print(f"{copy_session_name}...")
        print()

        shutil.copytree(
            f"{original_dir}/{session_name}",
            f"{original_dir}/{copy_session_name}",
            dirs_exist_ok=True,
        )

        # load original audio file
        wave_file = [
            f for f in os.listdir(f"{original_dir}/{copy_session_name}") if f.endswith(".wav")
        ][0]
        wave, sr = librosa.load(f"{original_dir}/{copy_session_name}/{wave_file}", sr=None)

        # load background and anomaly audio files (resample to match original sr)
        material = config.get("material")
        material_dir = material.get("dir")
        background_config = material.get("background")
        anomaly_config = material.get("anomaly")
        wave_background, _ = librosa.load(
            f"{material_dir}/{background_config.get('file_name')}", sr=sr
        )
        wave_anomaly, _ = librosa.load(
            f"{material_dir}/{anomaly_config.get('file_name')}", sr=sr
        )
        wave_anomaly = wave_anomaly[-sr * 6:]
        padding = len(wave) - len(wave_anomaly)
        wave_anomaly = np.concatenate(
            [np.zeros(padding // 2), wave_anomaly, np.zeros(padding // 2)], axis=0
        )

        # mix background and anomaly noise into original audio
        wave = (
            wave * original_config.get("weight")
            + wave_background[: len(wave)] * background_config.get("weight")
            + wave_anomaly * anomaly_config.get("weight")
        )
        sf.write(f"{original_dir}/{copy_session_name}/{wave_file}", wave, sr)

    print("All done.")


if __name__ == "__main__":
    main()
