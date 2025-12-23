import sys
import types
import torch

class _DummyProgressBar:
    def __init__(self, total):
        self.total = total
        self.updated = 0

    def update(self, amount):
        self.updated += amount


def _ensure_comfy_stubs():
    if "comfy" in sys.modules:
        return

    comfy_module = types.ModuleType("comfy")

    model_management_module = types.ModuleType("comfy.model_management")
    model_management_module.get_torch_device = lambda: torch.device("cpu")
    model_management_module.unet_dtype = lambda: torch.float32
    model_management_module.load_models_gpu = lambda models: None

    utils_module = types.ModuleType("comfy.utils")
    utils_module.ProgressBar = _DummyProgressBar

    class _AnyType(str):
        def __new__(cls, name="*"):
            return str.__new__(cls, name)

        def __ne__(self, other):
            return False

    def _any_type(name="*"):
        return _AnyType(name)

    utils_module.AnyType = _AnyType
    utils_module.any_type = _any_type

    samplers_module = types.ModuleType("comfy.samplers")

    class _DummyKSampler:
        def __init__(self, model, steps, device=None):
            self.model = model
            self.steps = steps
            self.device = device or torch.device("cpu")
            self.sigmas = torch.linspace(1.0, 0.0, steps)

        def sample(self, *args, **kwargs):
            return torch.zeros((1, 16, 1, 8, 8), device=self.device)

    samplers_module.KSampler = _DummyKSampler

    comfy_module.model_management = model_management_module
    comfy_module.utils = utils_module
    comfy_module.samplers = samplers_module

    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.model_management"] = model_management_module
    sys.modules["comfy.utils"] = utils_module
    sys.modules["comfy.samplers"] = samplers_module


_ensure_comfy_stubs()
