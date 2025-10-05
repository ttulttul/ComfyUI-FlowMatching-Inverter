import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "__init__.py"

FLOW_MATCHING_CHANNELS = 16
FLOW_MATCHING_TEMPORAL = 1
FLOW_MATCHING_WIDTH = 8
FLOW_MATCHING_HEIGHT = 8


def load_module():
    spec = importlib.util.spec_from_file_location("flowmatching_inverter", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_flow_matching_latent(batch_size=1, width=FLOW_MATCHING_WIDTH, height=FLOW_MATCHING_HEIGHT):
    samples = torch.randn(batch_size, FLOW_MATCHING_CHANNELS, FLOW_MATCHING_TEMPORAL, width, height)
    return {"samples": samples}


def assert_flow_matching_shape(tensor, batch_size=1, width=FLOW_MATCHING_WIDTH, height=FLOW_MATCHING_HEIGHT):
    expected = (batch_size, FLOW_MATCHING_CHANNELS, FLOW_MATCHING_TEMPORAL, width, height)
    assert tensor.shape == expected


def test_latent_gaussian_blur_modifies_values():
    module = load_module()
    node = module.LatentGaussianBlur()
    latent = make_flow_matching_latent()

    (result,) = node.blur_latent(latent, sigma=1.5, blur_mode="Spatial Only")

    original = latent["samples"]
    blurred = result["samples"]

    assert_flow_matching_shape(original)
    assert_flow_matching_shape(blurred)
    assert not torch.allclose(blurred, original)


def test_latent_add_noise_reproducible_with_seed():
    module = load_module()
    node = module.LatentAddNoise()
    latent = make_flow_matching_latent()

    (first,) = node.add_noise(latent, seed=123, strength=0.8)
    (second,) = node.add_noise(latent, seed=123, strength=0.8)
    (third,) = node.add_noise(latent, seed=321, strength=0.8)

    assert_flow_matching_shape(first["samples"])
    assert_flow_matching_shape(second["samples"])
    assert_flow_matching_shape(third["samples"])
    assert torch.allclose(first["samples"], second["samples"])
    assert not torch.allclose(first["samples"], third["samples"])


def test_perlin_noise_strength_affects_latent():
    module = load_module()
    node = module.LatentPerlinFractalNoise()
    latent = make_flow_matching_latent()

    (result,) = node.add_perlin_noise(
        latent,
        seed=7,
        frequency=1.5,
        octaves=2,
        persistence=0.5,
        lacunarity=2.0,
        strength=0.5,
        channel_mode="shared",
    )

    assert_flow_matching_shape(result["samples"])
    assert not torch.allclose(result["samples"], latent["samples"])


def test_simplex_noise_zero_strength_is_noop():
    module = load_module()
    node = module.LatentSimplexNoise()
    latent = make_flow_matching_latent()

    (result,) = node.add_simplex_noise(
        latent,
        seed=42,
        frequency=1.0,
        octaves=2,
        persistence=0.5,
        lacunarity=2.0,
        strength=0.0,
        channel_mode="shared",
        temporal_mode="locked",
    )

    assert_flow_matching_shape(result["samples"])
    assert torch.allclose(result["samples"], latent["samples"])


def test_conditioning_nodes_modify_embeddings_and_metadata():
    module = load_module()
    noise_node = module.ConditioningAddNoise()
    blur_node = module.ConditioningGaussianBlur()

    embedding = torch.randn(2, 4, 6)
    pooled = torch.randn(4)
    conditioning = [[embedding.clone(), {"pooled_output": pooled.clone()}]]

    (noised,) = noise_node.add_noise(conditioning, seed=0, strength=0.5)
    (blurred,) = blur_node.blur(noised, sigma=1.0)

    noised_embedding = noised[0][0]
    noised_pooled = noised[0][1]["pooled_output"]

    assert not torch.allclose(noised_embedding, embedding)
    assert not torch.allclose(noised_pooled, pooled)

    blurred_embedding = blurred[0][0]
    assert blurred_embedding.shape == embedding.shape
