import importlib.util
from pathlib import Path

import pytest
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


def make_flow_matching_latent(
    batch_size=1,
    temporal=FLOW_MATCHING_TEMPORAL,
    width=FLOW_MATCHING_WIDTH,
    height=FLOW_MATCHING_HEIGHT,
):
    samples = torch.randn(batch_size, FLOW_MATCHING_CHANNELS, temporal, width, height)
    return {"samples": samples}


def assert_flow_matching_shape(
    tensor,
    batch_size=1,
    temporal=FLOW_MATCHING_TEMPORAL,
    width=FLOW_MATCHING_WIDTH,
    height=FLOW_MATCHING_HEIGHT,
):
    expected = (batch_size, FLOW_MATCHING_CHANNELS, temporal, width, height)
    assert tensor.shape == expected


def make_latent_from_tensor(tensor):
    return {"samples": tensor}


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


def test_latent_gaussian_blur_rejects_rank2_tensor():
    module = load_module()
    node = module.LatentGaussianBlur()
    malformed = make_latent_from_tensor(torch.randn(16, 16))

    with pytest.raises((RuntimeError, IndexError)):
        node.blur_latent(malformed, sigma=1.0, blur_mode="Spatial Only")


def test_latent_gaussian_blur_zero_sigma_returns_original_object():
    module = load_module()
    node = module.LatentGaussianBlur()
    latent = make_flow_matching_latent()

    (result,) = node.blur_latent(latent, sigma=0.0, blur_mode="Spatial Only")

    assert result is latent


def test_latent_gaussian_blur_handles_video_latent():
    module = load_module()
    node = module.LatentGaussianBlur()
    video = make_flow_matching_latent(temporal=3, width=4, height=4)

    (result,) = node.blur_latent(video, sigma=1.0, blur_mode="Spatial Only")

    original = video["samples"]
    blurred = result["samples"]

    assert_flow_matching_shape(blurred, temporal=3, width=4, height=4)
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


def test_latent_add_noise_negative_strength_inverts_delta():
    module = load_module()
    node = module.LatentAddNoise()
    latent = make_flow_matching_latent()

    (positive,) = node.add_noise(latent, seed=42, strength=0.75)
    (negative,) = node.add_noise(latent, seed=42, strength=-0.75)

    positive_delta = positive["samples"] - latent["samples"]
    negative_delta = negative["samples"] - latent["samples"]

    assert torch.allclose(negative_delta, -positive_delta, atol=1e-6)


def test_latent_add_noise_with_zero_variance_latent_is_stable():
    module = load_module()
    node = module.LatentAddNoise()
    zero_latent = make_latent_from_tensor(torch.zeros(1, FLOW_MATCHING_CHANNELS, FLOW_MATCHING_TEMPORAL, FLOW_MATCHING_WIDTH, FLOW_MATCHING_HEIGHT))

    (result,) = node.add_noise(zero_latent, seed=0, strength=50.0)

    assert torch.allclose(result["samples"], zero_latent["samples"])


def test_latent_add_noise_rejects_integer_latent():
    module = load_module()
    node = module.LatentAddNoise()
    latent = make_latent_from_tensor(
        torch.zeros(
            1,
            FLOW_MATCHING_CHANNELS,
            FLOW_MATCHING_TEMPORAL,
            FLOW_MATCHING_WIDTH,
            FLOW_MATCHING_HEIGHT,
            dtype=torch.int32,
        )
    )

    with pytest.raises(RuntimeError):
        node.add_noise(latent, seed=0, strength=1.0)


def test_latent_frequency_split_zero_sigma_returns_zero_high_pass():
    module = load_module()
    node = module.LatentFrequencySplit()
    constant = torch.ones(1, FLOW_MATCHING_CHANNELS, FLOW_MATCHING_TEMPORAL, 4, 4)
    latent = make_latent_from_tensor(constant)

    low, high = node.split(latent, sigma=0.0)

    assert_flow_matching_shape(low["samples"], width=4, height=4)
    assert_flow_matching_shape(high["samples"], width=4, height=4)
    assert torch.allclose(low["samples"], constant)
    assert torch.allclose(high["samples"], torch.zeros_like(constant))


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


def test_perlin_noise_negative_strength_inverts_delta():
    module = load_module()
    node = module.LatentPerlinFractalNoise()
    latent = make_flow_matching_latent()

    args = dict(
        seed=11,
        frequency=2.5,
        octaves=3,
        persistence=0.4,
        lacunarity=2.0,
    )

    (positive,) = node.add_perlin_noise(latent, strength=0.8, channel_mode="shared", **args)
    (negative,) = node.add_perlin_noise(latent, strength=-0.8, channel_mode="shared", **args)

    positive_delta = positive["samples"] - latent["samples"]
    negative_delta = negative["samples"] - latent["samples"]

    assert torch.allclose(negative_delta, -positive_delta, atol=1e-6)


def test_perlin_noise_per_channel_channels_diverge():
    module = load_module()
    node = module.LatentPerlinFractalNoise()
    latent_tensor = torch.randn(1, 4, 1, 8, 8)
    latent = make_latent_from_tensor(latent_tensor.clone())

    (result,) = node.add_perlin_noise(
        latent,
        seed=3,
        frequency=1.5,
        octaves=2,
        persistence=0.4,
        lacunarity=2.0,
        strength=0.75,
        channel_mode="per_channel",
    )

    samples = result["samples"]
    assert not torch.isnan(samples).any()
    deltas = samples - latent_tensor
    assert not torch.allclose(deltas[0, 0], deltas[0, 1])


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


def test_simplex_noise_negative_strength_inverts_delta():
    module = load_module()
    node = module.LatentSimplexNoise()
    latent = make_flow_matching_latent()

    args = dict(
        seed=9,
        frequency=1.0,
        octaves=2,
        persistence=0.3,
        lacunarity=2.0,
        channel_mode="per_channel",
        temporal_mode="animated",
    )

    (positive,) = node.add_simplex_noise(latent, strength=0.5, **args)
    (negative,) = node.add_simplex_noise(latent, strength=-0.5, **args)

    positive_delta = positive["samples"] - latent["samples"]
    negative_delta = negative["samples"] - latent["samples"]

    assert torch.allclose(negative_delta, -positive_delta, atol=1e-6)


def test_fbm_noise_zero_strength_returns_original_object():
    module = load_module()
    node = module.LatentFractalBrownianMotion()
    latent = make_flow_matching_latent()

    (result,) = node.add_fbm_noise(
        latent,
        seed=0,
        base_noise="simplex",
        frequency=1.0,
        feature_points=8,
        octaves=2,
        persistence=0.5,
        lacunarity=2.0,
        distance_metric="euclidean",
        jitter=0.25,
        strength=0.0,
        channel_mode="shared",
        temporal_mode="locked",
    )

    assert result is not latent
    assert torch.allclose(result["samples"], latent["samples"])


def test_fbm_noise_with_constant_input_generates_variation():
    module = load_module()
    node = module.LatentFractalBrownianMotion()
    latent = make_latent_from_tensor(torch.zeros(1, 4, 1, 10, 10))

    (result,) = node.add_fbm_noise(
        latent,
        seed=1,
        base_noise="perlin",
        frequency=1.5,
        feature_points=4,
        octaves=3,
        persistence=0.45,
        lacunarity=2.2,
        distance_metric="euclidean",
        jitter=0.1,
        strength=0.8,
        channel_mode="per_channel",
        temporal_mode="locked",
    )

    delta = result["samples"] - latent["samples"]
    assert not torch.allclose(delta, torch.zeros_like(delta))


def test_fbm_noise_animated_temporal_frames_differ():
    module = load_module()
    node = module.LatentFractalBrownianMotion()
    latent = make_flow_matching_latent(temporal=2, width=6, height=6)

    (result,) = node.add_fbm_noise(
        latent,
        seed=7,
        base_noise="simplex",
        frequency=1.0,
        feature_points=8,
        octaves=2,
        persistence=0.6,
        lacunarity=2.0,
        distance_metric="euclidean",
        jitter=0.3,
        strength=0.9,
        channel_mode="shared",
        temporal_mode="animated",
    )

    samples = result["samples"]
    assert not torch.allclose(samples[0, :, 0], samples[0, :, 1])


def test_reaction_diffusion_per_channel_channels_diverge():
    module = load_module()
    node = module.LatentReactionDiffusion()
    latent = make_latent_from_tensor(torch.zeros(1, 3, 1, 12, 12))

    (result,) = node.add_reaction_diffusion(
        latent,
        seed=5,
        iterations=5,
        feed_rate=0.03,
        kill_rate=0.058,
        diffusion_u=0.16,
        diffusion_v=0.08,
        time_step=1.0,
        strength=1.0,
        channel_mode="per_channel",
        temporal_mode="locked",
    )

    samples = result["samples"]
    assert not torch.isnan(samples).any()
    assert not torch.allclose(samples[0, 0], samples[0, 1])


def test_swirl_noise_channel_fraction_zero_is_noop():
    module = load_module()
    node = module.LatentSwirlNoise()
    latent = make_flow_matching_latent(width=6, height=6)

    (result,) = node.add_swirl_noise(
        latent,
        seed=9,
        vortices=3,
        channel_mode="global",
        channel_fraction=0.0,
        strength=1.0,
        radius=0.5,
        center_spread=0.25,
        direction_bias=0.0,
        mix=1.0,
    )

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


def test_conditioning_add_noise_negative_strength_reverses_delta():
    module = load_module()
    node = module.ConditioningAddNoise()

    embedding = torch.randn(3, 2, 5)
    pooled = torch.randn(5)
    conditioning = [[embedding.clone(), {"pooled_output": pooled.clone()}]]

    (positive,) = node.add_noise(conditioning, seed=123, strength=0.6)
    (negative,) = node.add_noise(conditioning, seed=123, strength=-0.6)

    positive_embed_delta = positive[0][0] - embedding
    negative_embed_delta = negative[0][0] - embedding
    assert torch.allclose(negative_embed_delta, -positive_embed_delta, atol=1e-6)

    positive_pooled_delta = positive[0][1]["pooled_output"] - pooled
    negative_pooled_delta = negative[0][1]["pooled_output"] - pooled
    assert torch.allclose(negative_pooled_delta, -positive_pooled_delta, atol=1e-6)


def test_conditioning_add_noise_leaves_non_tensor_entries_untouched():
    module = load_module()
    node = module.ConditioningAddNoise()

    conditioning = [["not a tensor", {"pooled_output": "still not a tensor"}]]

    (result,) = node.add_noise(conditioning, seed=0, strength=1.0)

    assert result[0][0] == "not a tensor"
    assert result[0][1]["pooled_output"] == "still not a tensor"


def test_conditioning_gaussian_blur_zero_sigma_returns_original_list():
    module = load_module()
    node = module.ConditioningGaussianBlur()

    embedding = torch.randn(2, 4, 6)
    pooled = torch.randn(6)
    conditioning = [[embedding.clone(), {"pooled_output": pooled.clone()}]]

    (result,) = node.blur(conditioning, sigma=0.0)

    assert result is conditioning


def test_conditioning_frequency_split_handles_non_tensor_entries():
    module = load_module()
    node = module.ConditioningFrequencySplit()

    conditioning = [["prompt", {"pooled_output": "meta"}]]

    low, high = node.split(conditioning, sigma=0.5)

    assert low[0][0] == "prompt"
    assert high[0][0] == "prompt"
    assert low[0][1]["pooled_output"] == "meta"
    assert high[0][1]["pooled_output"] == "meta"


def test_conditioning_frequency_merge_mismatched_lengths_raises():
    module = load_module()
    node = module.ConditioningFrequencyMerge()

    low_pass = [[torch.zeros(2, 3), {}]]
    high_pass = [[torch.zeros(2, 3), {}], [torch.zeros(2, 3), {}]]

    with pytest.raises(ValueError):
        node.merge(low_pass, high_pass, low_gain=1.0, high_gain=1.0)
