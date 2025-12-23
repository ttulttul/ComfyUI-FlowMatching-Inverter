import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "__init__.py"


def load_module():
    spec = importlib.util.spec_from_file_location("flowmatching_inverter", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_node_mappings_only_expose_inverter_nodes():
    module = load_module()

    assert "QwenRectifiedFlowInverter" in module.NODE_CLASS_MAPPINGS
    assert "LatentHybridInverter" in module.NODE_CLASS_MAPPINGS
    assert "MemoryDiagnosticsPassThrough" in module.NODE_CLASS_MAPPINGS

    # moved to Skoogeer-Noise
    assert "LatentGaussianBlur" not in module.NODE_CLASS_MAPPINGS
    assert "LatentAddNoise" not in module.NODE_CLASS_MAPPINGS
    assert "ConditioningAddNoise" not in module.NODE_CLASS_MAPPINGS


def test_memory_diagnostics_pass_through_logs_and_returns_input(capsys):
    module = load_module()
    node = module.MemoryDiagnosticsPassThrough()

    payload = {"samples": torch.zeros(1)}

    (result,) = node.log_memory(payload, label="unit", synchronize="disable")

    assert result is payload

    output = capsys.readouterr().out

    assert "[MemoryDiagnostics]" in output
    assert "label=unit" in output
    assert "device=" in output
