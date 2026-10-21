"""从 ModelScope 下载 SAM3 权重到 pretrained_weights/sam3/facebook/sam3/sam3.pt。

正确仓库 id: facebook/sam3（AI-ModelScope/sam3 不存在）。

用法（在仓库根目录）::

    python pretrained_weights/download_sam3.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM3_PT = ROOT / "pretrained_weights" / "sam3" / "facebook" / "sam3" / "sam3.pt"
CACHE_DIR = ROOT / "pretrained_weights" / "sam3"


def ensure_sam3_checkpoint(dest: str | Path | None = None) -> Path:
    """若 ``dest`` 已存在则直接返回；否则用 ModelScope 下载 ``sam3.pt``。"""
    path = Path(dest) if dest is not None else DEFAULT_SAM3_PT
    if path.is_file():
        return path.resolve()

    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError as exc:
        raise ImportError(
            "自动下载 SAM3 需要 modelscope：pip install modelscope"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pretrained] SAM3 权重不存在，开始从 ModelScope 下载 → {path}")
    out = model_file_download(
        model_id="facebook/sam3",
        file_path="sam3.pt",
        cache_dir=str(CACHE_DIR),
        local_dir=str(path.parent),
    )
    downloaded = Path(out)
    if downloaded.resolve() != path.resolve() and downloaded.is_file():
        # ModelScope 可能写到同目录不同路径，统一到期望文件名
        if not path.is_file():
            path.write_bytes(downloaded.read_bytes())
    if not path.is_file():
        raise FileNotFoundError(f"SAM3 下载后仍未找到: {path}")
    print(f"[pretrained] SAM3 就绪: {path} ({path.stat().st_size / 1e9:.2f} GB)")
    return path.resolve()


def main() -> None:
    path = ensure_sam3_checkpoint(DEFAULT_SAM3_PT)
    print("sam3.pt:", path)
    print("size_GB:", round(path.stat().st_size / 1e9, 3))


if __name__ == "__main__":
    main()
