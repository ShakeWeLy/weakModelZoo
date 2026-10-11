"""VM-UNet 分割模型。"""
import importlib.util
import sys
from pathlib import Path

_vm_unet_path = Path(__file__).parent / "VM_UNet.py"
_vm_unet_spec = importlib.util.spec_from_file_location(
    "src.models.segment.vm_unet.model", _vm_unet_path
)
_vm_unet_module = importlib.util.module_from_spec(_vm_unet_spec)
sys.modules[_vm_unet_spec.name] = _vm_unet_module
assert _vm_unet_spec.loader is not None
_vm_unet_spec.loader.exec_module(_vm_unet_module)
VMUNet = _vm_unet_module.VMUNet

__all__ = ["VMUNet"]
