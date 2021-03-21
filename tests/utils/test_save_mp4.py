from pathlib import Path

import pytest

import numpy as np

from dail.utils import save_mp4


def test_should_save_mp4_file(tmp_path: Path) -> None:
    frames = []
    for _ in range(60):
        frames.append(np.random.randint(low=0, high=255, size=(10, 10, 3)))

    file_path = tmp_path / "temp.mp4"

    save_mp4(frames, file_path)

    assert file_path.is_file()


def test_should_raise_an_error(tmp_path: Path) -> None:
    frames = []
    file_path = tmp_path / "temp"

    expected_error_message = (
        f"Failed to save frames to file: {file_path}. Only support .mp4 file format."
    )

    with pytest.raises(ValueError, match=expected_error_message):
        save_mp4(frames, file_path)
