import os
import yaml
import shutil
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

def main(config_path: str = "configs/audio_processing.yaml") -> None:
    """
    Copy original video and preprocess auido file with backgournd and annomaly noise.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # copy original video directory
    original_config = config.get("original")
    original_dir = original_config.get("dir")
    video_names = original_config.get("video_names")
    print("Preprocessing fllowing videos.")
    print(video_names)
    print()

    for video_name in video_names:
        copy_video_name = video_name + "_audio"
        print(f"{copy_video_name}...")
        print()

        shutil.copytree(f"{original_dir}/{video_name}", f"{original_dir}/{copy_video_name}", dirs_exist_ok=True)
        del video_name

        # load original audio file
        wave_file = [f for f in os.listdir(f"{original_dir}/{copy_video_name}") if f.endswith(".wav")][0]
        wave, sr = librosa.load(f"{original_dir}/{copy_video_name}/{wave_file}")

        # load background and annormaly audio file
        material = config.get("material")
        material_dir = material.get("dir")
        background_config = material.get("background")
        annormaly_config = material.get("annormaly")
        wave_background, sr_background = librosa.load(f"{material_dir}/{background_config.get('file_name')}")
        wave_annormaly, sr_annormaly = librosa.load(f"{material_dir}/{annormaly_config.get('file_name')}")
        wave_annormaly = wave_annormaly[-sr_annormaly*6:]
        padding = len(wave) - len(wave_annormaly)
        wave_annormaly = np.concatenate([np.zeros(padding//2), wave_annormaly, np.zeros(padding//2)], axis=0)

        # adding background and annormaly noise
        wave = wave*original_config.get("weight") \
                + wave_background[:len(wave)]*background_config.get("weight") \
                + wave_annormaly*annormaly_config.get("weight")
        sf.write(f"{original_dir}/{copy_video_name}/{wave_file}", wave, sr)
    print("All done.")
    return


if __name__ == "__main__":
    main()