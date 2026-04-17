from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

AVIF_EXTENSIONS = {".avif", ".heif", ".heic"}

try:  # pragma: no cover - optional dependency
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional dependency
    register_heif_opener = None
else:  # pragma: no cover - exercised only when pillow-heif is installed
    register_heif_opener()


def load_image_rgb(image_path: str | os.PathLike[str]) -> Image.Image:
    path = Path(image_path)
    try:
        with Image.open(path) as image_file:
            return image_file.convert("RGB")
    except UnidentifiedImageError as exc:
        if path.suffix.lower() not in AVIF_EXTENSIONS:
            raise
        return _load_avif_with_fallback(path, exc)


def probe_image_readable(image_path: str | os.PathLike[str]) -> tuple[bool, str]:
    try:
        image = load_image_rgb(image_path)
        image.close()
        return True, ""
    except Exception as exc:  # pragma: no cover - error formatting only
        return False, f"{type(exc).__name__}: {exc}"


def _load_avif_with_fallback(path: Path, original_exc: Exception) -> Image.Image:
    if sys.platform == "darwin":
        sips_binary = shutil.which("sips")
        if sips_binary:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                subprocess.run(
                    [sips_binary, "-s", "format", "png", str(path), "--out", str(temp_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                with Image.open(temp_path) as image_file:
                    return image_file.convert("RGB")
            except Exception:
                pass
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

    raise UnidentifiedImageError(
        f"cannot identify image file '{path}'. "
        "Install pillow-heif or use a system decoder that supports AVIF."
    ) from original_exc
