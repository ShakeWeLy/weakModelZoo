"""PSC-UNet 分割模型。"""
import importlib.util
import sys
from pathlib import Path

_psc_unet_path = Path(__file__).parent / "PSC_UNet.py"
_psc_unet_spec = importlib.util.spec_from_file_location(
    "src.models.segment.psc_unet.model", _psc_unet_path
)
_psc_unet_module = importlib.util.module_from_spec(_psc_unet_spec)
sys.modules[_psc_unet_spec.name] = _psc_unet_module
assert _psc_unet_spec.loader is not None
_psc_unet_spec.loader.exec_module(_psc_unet_module)
PSC_UNet = _psc_unet_module.PSC_UNet

__all__ = ["PSC_UNet"]
