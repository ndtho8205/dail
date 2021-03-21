from typing import List

from pathlib import Path

import numpy as np
import imageio


def save_mp4(frames: List[np.ndarray], file_path: Path) -> None:
    """Save a list of frames as a .mp4 file."""
    if file_path.suffix != ".mp4":
        raise ValueError(
            f"Failed to save frames to file: {file_path}. Only support .mp4 file format."
        )

    frames = np.asarray(frames)
    imageio.mimwrite(file_path, frames, fps=60)
