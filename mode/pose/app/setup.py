#!/usr/bin/env python3

import sys
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


MODELS = {
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/"
        "mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/"
        "pose_landmarker_full.task",

    "hand_landmarker.task":
        "https://storage.googleapis.com/"
        "mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/"
        "hand_landmarker.task",
}


def download_model(filename, url):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = MODEL_DIR / filename

    if destination.exists():

        print(
            f"[OK] {filename} already exists"
        )

        return

    print(
        f"[DOWNLOAD] {filename}"
    )

    print(
        f"           {url}"
    )

    try:

        urllib.request.urlretrieve(
            url,
            destination
        )

    except Exception as exc:

        if destination.exists():
            destination.unlink()

        print(
            f"[ERROR] Could not download {filename}"
        )

        print(
            f"        {exc}"
        )

        sys.exit(1)

    print(
        f"[OK] Saved to {destination}"
    )


def main():

    print()
    print(
        "MediaPipe Robot Arm Teleoperation"
    )
    print(
        "Downloading required models..."
    )
    print()

    for filename, url in MODELS.items():

        download_model(
            filename,
            url
        )

    print()
    print("Models ready.")
    print()
    print("Run:")
    print()
    print("    python teleop.py")
    print()


if __name__ == "__main__":
    main()