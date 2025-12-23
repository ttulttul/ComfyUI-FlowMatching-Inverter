import comfy.model_management
import comfy.utils

import torch
import torch.nn.functional as F


def slerp(val, low, high):
    """
    Robust, batch-aware Spherical Linear Interpolation for PyTorch tensors.
    """
    dims = low.shape

    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = F.normalize(low, p=2, dim=1)
    high_norm = F.normalize(high, p=2, dim=1)

    low_norm[torch.isnan(low_norm)] = 0.0
    high_norm[torch.isnan(high_norm)] = 0.0

    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)

    cond = so > 1e-3

    res = torch.zeros_like(low)

    if cond.any():
        omega_cond = omega[cond].unsqueeze(1)
        so_cond = so[cond].unsqueeze(1)

        term1 = (torch.sin((1.0 - val) * omega_cond) / so_cond) * low[cond]
        term2 = (torch.sin(val * omega_cond) / so_cond) * high[cond]
        res[cond] = term1 + term2

    if not cond.all():
        res[~cond] = (1.0 - val) * low[~cond] + val * high[~cond]
    return res.reshape(dims)


def _add_scheduled_noise(*, model, latent, seed, steps, noise_strength):
    if noise_strength == 0.0:
        return latent

    device = comfy.model_management.get_torch_device()
    latent_tensor = latent["samples"].clone().to(device)

    import comfy.samplers  # imported lazily to match ComfyUI load order

    sampler = comfy.samplers.KSampler(model, steps=steps, device=device)
    sigmas = sampler.sigmas

    start_step = steps - int(steps * noise_strength)
    if start_step >= len(sigmas):
        start_step = len(sigmas) - 1

    sigma = sigmas[start_step].to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(latent_tensor.shape, generator=generator, device=device, dtype=latent_tensor.dtype)

    noised_latent = latent_tensor + noise * sigma

    out = latent.copy()
    out["samples"] = noised_latent.cpu()
    return out


class LatentHybridInverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "UNet model used for both inversion and diffusion passes."}),
                "latent_image": ("LATENT", {"tooltip": "Latent tensor to invert and re-noise."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning that guides both passes."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000, "tooltip": "Solver steps executed for inversion and forward diffusion."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength for both inversion and forward diffusion. The target noise level."}),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "0.0 = 100% Forward Diffusion (Stable), 1.0 = 100% Inverter (Creative)"}),
                "inverter_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the creative, perturbed inversion process."}),
                "forward_diffusion_seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the stable, 'perfect' noise anchor."}),
                "velocity_amplification": ("FLOAT", {"default": 1.0, "min": -0.9, "max": 2.0, "step": 0.05, "tooltip": "Deterministically scales the predicted velocity each step."}),
                "velocity_perturb_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Random velocity noise level for the inverter stage."}),
                "internal_precision": (["bfloat16", "float16", "force_float32"], {"default": "bfloat16", "tooltip": "Precision used for inverter UNet autocast."}),
                "normalize_output": (["enable", "disable"], {"default": "enable", "tooltip": "Normalizing the inverter output is highly recommended."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "hybrid_invert"
    CATEGORY = "Qwen/Sampling"

    def hybrid_invert(
        self,
        model,
        latent_image,
        positive,
        steps,
        strength,
        blend_factor,
        inverter_seed,
        forward_diffusion_seed,
        velocity_amplification,
        velocity_perturb_strength,
        internal_precision,
        normalize_output,
    ):
        if strength == 0.0:
            return (latent_image,)

        device = comfy.model_management.get_torch_device()
        native_dtype = comfy.model_management.unet_dtype()

        inverter_node = QwenRectifiedFlowInverter()
        inverted_latent, = inverter_node.invert(
            model=model,
            latent_image=latent_image,
            positive=positive,
            seed=inverter_seed,
            steps=steps,
            inversion_strength=strength,
            velocity_amplification=velocity_amplification,
            velocity_perturb_strength=velocity_perturb_strength,
            internal_precision=internal_precision,
            normalize_output=normalize_output,
        )

        forward_latent = _add_scheduled_noise(
            model=model,
            latent=latent_image,
            seed=forward_diffusion_seed,
            steps=steps,
            noise_strength=strength,
        )

        inverted_samples = inverted_latent["samples"].to(device, native_dtype)
        forward_samples = forward_latent["samples"].to(device, native_dtype)

        if inverted_samples.dim() == 5 and forward_samples.dim() == 4:
            forward_samples = forward_samples.unsqueeze(2)
        elif inverted_samples.dim() == 4 and forward_samples.dim() == 5:
            inverted_samples = inverted_samples.unsqueeze(2)

        if blend_factor == 0.0:
            final_latent = forward_samples
        elif blend_factor == 1.0:
            final_latent = inverted_samples
        else:
            final_latent = slerp(blend_factor, forward_samples, inverted_samples)

        out = latent_image.copy()
        out["samples"] = final_latent.cpu()
        return (out, )


class QwenRectifiedFlowInverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "UNet to integrate during the rectified flow inversion."}),
                "latent_image": ("LATENT", {"tooltip": "Latent to be inverted toward the clean state."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning applied during inversion."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the random velocity perturbation."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000, "tooltip": "Number of integration steps available for the inverter."}),
                "inversion_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Fraction of the schedule to walk backward during inversion."}),
                "velocity_amplification": ("FLOAT", {
                    "default": 0.0,
                    "min": -0.9,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Deterministically amplifies (>0) or dampens (<0) the predicted velocity at each step."
                }),
                "velocity_perturb_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Adds seeded random noise to the velocity. >1.0 will heavily randomize the path."
                }),
                "internal_precision": (["bfloat16", "float16", "force_float32"], {"default": "bfloat16", "tooltip": "Autocast precision used while running the inverter UNet."}),
                "normalize_output": (["enable", "disable"], {"default": "enable", "tooltip": "Crucial for high inversion strengths. Rescales the output latent to a standard distribution (mean=0, std=1) to prevent the sampler from collapsing."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "invert"
    CATEGORY = "Qwen/Sampling"

    def invert(
        self,
        model,
        latent_image,
        positive,
        seed,
        steps,
        inversion_strength,
        velocity_amplification,
        velocity_perturb_strength,
        internal_precision,
        normalize_output,
    ):
        if inversion_strength == 0.0:
            return (latent_image,)

        device = comfy.model_management.get_torch_device()
        native_dtype = comfy.model_management.unet_dtype()

        autocast_dtype = (
            torch.bfloat16
            if internal_precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
            if internal_precision == "force_float32"
            else torch.float16
        )

        clean_latent = latent_image["samples"].clone().to(device, native_dtype)
        context = positive[0][0].to(device, native_dtype)

        qwen_transformer = model.model.diffusion_model
        comfy.model_management.load_models_gpu([model])

        inversion_steps = max(1, int(steps * inversion_strength))
        end_t = inversion_strength
        timesteps = torch.linspace(0, end_t, inversion_steps + 1, device=device, dtype=native_dtype)

        x_t = clean_latent.unsqueeze(2) if clean_latent.dim() == 4 else clean_latent

        pbar = comfy.utils.ProgressBar(inversion_steps)

        with torch.no_grad():
            for i in range(inversion_steps):
                t_current = timesteps[i]
                t_next = timesteps[i + 1]
                t_current_b = torch.full((x_t.shape[0],), t_current, device=device, dtype=native_dtype)

                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    predicted_velocity = qwen_transformer(x=x_t, timestep=t_current_b, context=context, attention_mask=None)

                if predicted_velocity.dim() == 4:
                    predicted_velocity = predicted_velocity.unsqueeze(2)

                if velocity_amplification != 0.0:
                    predicted_velocity = predicted_velocity * (1.0 + velocity_amplification)

                if velocity_perturb_strength > 0:
                    step_seed = seed + i
                    generator = torch.Generator(device=device).manual_seed(step_seed)

                    velocity_std = torch.std(predicted_velocity)

                    noise = torch.randn(
                        predicted_velocity.shape,
                        generator=generator,
                        device=device,
                        dtype=native_dtype,
                    )

                    perturbation = noise * velocity_std * velocity_perturb_strength
                    predicted_velocity = predicted_velocity + perturbation

                dt = t_next - t_current
                x_t = x_t + predicted_velocity * dt

                if torch.isnan(x_t).any():
                    raise RuntimeError("NaN detected at step %s. Increase the precision setting." % (i + 1))

                pbar.update(1)

        if normalize_output == "enable":
            for i in range(x_t.shape[0]):
                latent_item = x_t[i]
                mean = torch.mean(latent_item)
                std = torch.std(latent_item)
                if std > 1e-5:
                    x_t[i] = (latent_item - mean) / std
                else:
                    x_t[i] = latent_item - mean

        out = latent_image.copy()
        out["samples"] = x_t.cpu()
        return (out, )


def _format_memory(value):
    if value is None:
        return "n/a"
    return f"{value / (1024 ** 2):.2f} MiB"


class _WildcardType(str):
    def __new__(cls, name="*"):
        return super().__new__(cls, name)

    def __ne__(self, other):
        return False


def _resolve_any_type(name):
    utils = getattr(comfy, "utils", None)
    constructor = None
    if utils is not None:
        constructor = getattr(utils, "AnyType", None)
        if constructor is None:
            constructor = getattr(utils, "any_type", None)

    if constructor is not None:
        try:
            return constructor(name)
        except TypeError:
            try:
                return constructor()
            except TypeError:
                pass

    return _WildcardType(name)


def _collect_memory_stats(device, *, synchronize=False):
    stats = {
        "device": str(device),
        "backend": getattr(device, "type", "unknown"),
        "sync": synchronize,
    }

    if stats["backend"] == "cuda" and torch.cuda.is_available():
        try:
            if synchronize:
                torch.cuda.synchronize(device)

            free_bytes, total_bytes = torch.cuda.mem_get_info(device)

            stats.update({
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
                "max_allocated": torch.cuda.max_memory_allocated(device),
                "max_reserved": torch.cuda.max_memory_reserved(device),
                "free": free_bytes,
                "total": total_bytes,
            })
        except RuntimeError as exc:
            stats["error"] = f"CUDA stats unavailable: {exc}"
    elif stats["backend"] == "mps" and hasattr(torch, "mps"):
        try:
            if synchronize and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

            allocated = torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else None
            reserved = torch.mps.driver_allocated_memory() if hasattr(torch.mps, "driver_allocated_memory") else None

            stats.update({
                "allocated": allocated,
                "reserved": reserved,
            })

            if allocated is None and reserved is None:
                stats.setdefault("note", "MPS memory counters unavailable")
        except RuntimeError as exc:
            stats["error"] = f"MPS stats unavailable: {exc}"
    else:
        stats["note"] = "No GPU backend available"

    return stats


class MemoryDiagnosticsPassThrough:
    """Passes inputs through unchanged while reporting GPU memory usage."""

    _ANY = _resolve_any_type("memory_data")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (cls._ANY, {"tooltip": "Input to pass through unchanged."}),
            },
            "optional": {
                "label": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional label shown in the log."}),
                "synchronize": (["enable", "disable"], {"default": "enable", "tooltip": "Synchronize GPU kernels before sampling memory."}),
            },
        }

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("data",)
    FUNCTION = "log_memory"
    CATEGORY = "Qwen/Diagnostics"

    def log_memory(self, data, label="", synchronize="enable"):
        device = comfy.model_management.get_torch_device()
        sync = synchronize == "enable"

        stats = _collect_memory_stats(device, synchronize=sync)
        parts = ["[MemoryDiagnostics]", f"device={stats['device']}"]

        if label:
            parts.append(f"label={label}")

        if "error" in stats:
            parts.append(stats["error"])
        else:
            for key in ("allocated", "reserved", "max_allocated", "max_reserved", "free", "total"):
                value = stats.get(key)
                if value is not None:
                    parts.append(f"{key}={_format_memory(value)}")

            if "note" in stats:
                parts.append(stats["note"])

        print(" ".join(parts))

        return (data,)


NODE_CLASS_MAPPINGS = {
    "QwenRectifiedFlowInverter": QwenRectifiedFlowInverter,
    "LatentHybridInverter": LatentHybridInverter,
    "MemoryDiagnosticsPassThrough": MemoryDiagnosticsPassThrough,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenRectifiedFlowInverter": "Qwen Rectified Flow Inverter",
    "LatentHybridInverter": "Latent Hybrid Inverter (Qwen)",
    "MemoryDiagnosticsPassThrough": "Memory Diagnostics (Pass-Through)",
}

