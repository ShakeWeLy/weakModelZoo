import importlib.util
import sys
from pathlib import Path

_swin_unet_path = Path(__file__).parent / "Swin-UNet.py"
_swin_unet_spec = importlib.util.spec_from_file_location(
    "src.models.segment.swin_unet.model", _swin_unet_path
)
_swin_unet_module = importlib.util.module_from_spec(_swin_unet_spec)
sys.modules[_swin_unet_spec.name] = _swin_unet_module
assert _swin_unet_spec.loader is not None
_swin_unet_spec.loader.exec_module(_swin_unet_module)
SwinUNet = _swin_unet_module.SwinUNet

__all__ = ["SwinUNet"]
