import comfy.model_management
import comfy.utils

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def slerp(val, low, high):
    """
    Robust, batch-aware Spherical Linear Interpolation for PyTorch tensors.
    """
    dims = low.shape

    # Flatten per batch so we can SLERP across arbitrary shapes
    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = F.normalize(low, p=2, dim=1)
    high_norm = F.normalize(high, p=2, dim=1)

    # Guard against zero-length vectors
    low_norm[torch.isnan(low_norm)] = 0.0
    high_norm[torch.isnan(high_norm)] = 0.0

    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)

    # Fallback to lerp when the angle is tiny to avoid numerical issues
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

class LatentHybridInverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent_image": ("LATENT",),
                "positive": ("CONDITIONING", ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength for both inversion and forward diffusion. The target noise level."}),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "0.0 = 100% Forward Diffusion (Stable), 1.0 = 100% Inverter (Creative)"}),
                "inverter_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the creative, perturbed inversion process."}),
                "forward_diffusion_seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the stable, 'perfect' noise anchor."}),
                "velocity_amplification": ("FLOAT", {"default": 1.0, "min": -0.9, "max": 2.0, "step": 0.05}),
                "velocity_perturb_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.01}),
                "internal_precision": (["bfloat16", "float16", "force_float32"], {"default": "bfloat16"}),
                "normalize_output": (["enable", "disable"], {"default": "enable", "tooltip": "Normalizing the inverter output is highly recommended."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "hybrid_invert"
    CATEGORY = "Qwen/Sampling"

    def hybrid_invert(self, model, latent_image, positive, steps, strength,
                      blend_factor, inverter_seed, forward_diffusion_seed,
                      velocity_amplification, velocity_perturb_strength,
                      internal_precision, normalize_output):
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

        diffusion_node = LatentForwardDiffusion()
        forward_latent, = diffusion_node.add_scheduled_noise(
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
                "model": ("MODEL",),
                "latent_image": ("LATENT",),
                "positive": ("CONDITIONING", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the random velocity perturbation."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "inversion_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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
                "internal_precision": (["bfloat16", "float16", "force_float32"], {"default": "bfloat16"}),
                "normalize_output": (["enable", "disable"], {"default": "enable", "tooltip": "Crucial for high inversion strengths. Rescales the output latent to a standard distribution (mean=0, std=1) to prevent the sampler from collapsing."}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "invert"
    CATEGORY = "Qwen/Sampling"
        
    def invert(self, model, latent_image, positive, seed, steps, inversion_strength, velocity_amplification, velocity_perturb_strength, internal_precision, normalize_output):
        if inversion_strength == 0.0:
            return (latent_image,)

        device = comfy.model_management.get_torch_device()
        native_dtype = comfy.model_management.unet_dtype()

        autocast_dtype = torch.bfloat16 if internal_precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32 if internal_precision == "force_float32" else torch.float16

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
                t_next = timesteps[i+1]
                t_current_b = torch.full((x_t.shape[0],), t_current, device=device, dtype=native_dtype)
                
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    predicted_velocity = qwen_transformer(
                        x=x_t, timestep=t_current_b, context=context, attention_mask=None
                    )
                
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
                        dtype=native_dtype
                    )
                    
                    perturbation = noise * velocity_std * velocity_perturb_strength
                    predicted_velocity = predicted_velocity + perturbation

                dt = t_next - t_current
                x_t = x_t + predicted_velocity * dt

                if torch.isnan(x_t).any():
                    raise RuntimeError("NaN detected at step %s. Increase the precision setting." % (i+1))
                
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

class LatentGaussianBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "blur_mode": (["Spatial Only", "Spatial and Channel"], {"default": "Spatial Only"}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "blur_latent"
    CATEGORY = "Latent/Filter"

    def blur_latent(self, latent, sigma, blur_mode):
        if sigma == 0.0:
            return (latent,)

        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].clone().to(device)
        is_5d = latent_tensor.dim() == 5

        if is_5d:
            b, c, t, h, w = latent_tensor.shape
            latent_4d = latent_tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        else:
            latent_4d = latent_tensor

        if blur_mode == "Spatial Only":
            kernel_size = int(sigma * 6) + 1
            if kernel_size % 2 == 0: kernel_size += 1
            blurred_4d = TF.gaussian_blur(latent_4d, kernel_size=kernel_size, sigma=sigma)

        else:
            channels = latent_4d.shape[1]
            latent_5d_for_conv = latent_4d.unsqueeze(1)

            k_size_spatial = int(sigma * 6) + 1
            if k_size_spatial % 2 == 0: k_size_spatial += 1
            k_size_channel = min(channels, k_size_spatial)
            if k_size_channel % 2 == 0: k_size_channel -= 1
            if k_size_channel < 1: k_size_channel = 1

            ax_c = torch.linspace(-(k_size_channel - 1) / 2., (k_size_channel - 1) / 2., k_size_channel, device=device)
            ax_s = torch.linspace(-(k_size_spatial - 1) / 2., (k_size_spatial - 1) / 2., k_size_spatial, device=device)
            gauss_c = torch.exp(-0.5 * torch.square(ax_c / sigma))
            gauss_s = torch.exp(-0.5 * torch.square(ax_s / sigma))

            gauss_c = gauss_c.view(k_size_channel, 1, 1)
            gauss_h = gauss_s.view(1, k_size_spatial, 1)
            gauss_w = gauss_s.view(1, 1, k_size_spatial)
            kernel_3d = gauss_c * gauss_h * gauss_w

            kernel_3d /= torch.sum(kernel_3d)
            kernel_3d = kernel_3d.view(1, 1, k_size_channel, k_size_spatial, k_size_spatial)

            padding = (k_size_channel // 2, k_size_spatial // 2, k_size_spatial // 2)
            blurred_5d = F.conv3d(latent_5d_for_conv, kernel_3d, padding=padding)
            blurred_4d = blurred_5d.squeeze(1)

        if is_5d:
            blurred_tensor = blurred_4d.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        else:
            blurred_tensor = blurred_4d

        out = latent.copy()
        out["samples"] = blurred_tensor.cpu()
        return (out,)

class LatentAddNoise:
    """
    Adds a configurable amount of seeded random noise to a latent tensor.
    The strength is relative to the standard deviation of the input latent.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Strength of the noise. 1.0 adds noise with the same standard deviation as the latent."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_noise"
    CATEGORY = "Latent/Noise"

    def add_noise(self, latent, seed, strength):
        if strength == 0.0:
            return (latent,)

        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].clone().to(device)

        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            latent_tensor.shape,
            generator=generator,
            device=device,
            dtype=latent_tensor.dtype
        )

        latent_std = torch.std(latent_tensor)
        scaled_noise = noise * latent_std * strength

        noised_latent = latent_tensor + scaled_noise

        out = latent.copy()
        out["samples"] = noised_latent.cpu()

        return (out,)

class LatentForwardDiffusion:
    """
    Applies the 'natural' forward diffusion process to a clean latent.
    This produces a statistically 'perfect' noisy latent that samplers expect.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "noise_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "The point in the schedule to noise to. Must match the KSampler's effective start step."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_scheduled_noise"
    CATEGORY = "Latent/Noise"

    def add_scheduled_noise(self, model, latent, seed, steps, noise_strength):
        if noise_strength == 0.0:
            return (latent,)

        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].clone().to(device)

        sampler = comfy.samplers.KSampler(model, steps=steps, device=device)
        sigmas = sampler.sigmas

        start_step = steps - int(steps * noise_strength)
        if start_step >= len(sigmas):
            start_step = len(sigmas) - 1

        sigma = sigmas[start_step].to(device)

        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            latent_tensor.shape,
            generator=generator,
            device=device,
            dtype=latent_tensor.dtype
        )

        noised_latent = latent_tensor + noise * sigma

        out = latent.copy()
        out["samples"] = noised_latent.cpu()
        return (out,)


class ConditioningAddNoise:
    """Adds seeded Gaussian noise to conditioning embeddings and pooled outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "strength": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Noise strength relative to each tensor's standard deviation."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "add_noise"
    CATEGORY = "conditioning/noise"

    def add_noise(self, conditioning, seed, strength):
        if strength == 0.0:
            return (conditioning,)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = []

        for embedding, metadata in conditioning:
            if not isinstance(embedding, torch.Tensor):
                result.append([embedding, metadata])
                continue

            embedding_std = torch.std(embedding)
            noise = torch.randn(
                embedding.shape,
                generator=generator,
                device=embedding.device,
                dtype=embedding.dtype,
            )
            noised_embedding = embedding + noise * embedding_std * strength

            new_metadata = dict(metadata)
            pooled_output = new_metadata.get("pooled_output")

            if isinstance(pooled_output, torch.Tensor):
                pooled_std = torch.std(pooled_output)
                pooled_noise = torch.randn(
                    pooled_output.shape,
                    generator=generator,
                    device=pooled_output.device,
                    dtype=pooled_output.dtype,
                )
                new_metadata["pooled_output"] = pooled_output + pooled_noise * pooled_std * strength

            result.append([noised_embedding, new_metadata])

        return (result,)

NODE_CLASS_MAPPINGS = {
    "QwenRectifiedFlowInverter": QwenRectifiedFlowInverter,
    "LatentGaussianBlur": LatentGaussianBlur,
    "LatentAddNoise": LatentAddNoise,
    "LatentForwardDiffusion": LatentForwardDiffusion,
    "LatentHybridInverter": LatentHybridInverter,
    "ConditioningAddNoise": ConditioningAddNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenRectifiedFlowInverter": "Qwen Rectified Flow Inverter",
    "LatentGaussianBlur": "Latent Gaussian Blur",
    "LatentAddNoise": "Add Latent Noise (Seeded)",
    "LatentForwardDiffusion": "Forward Diffusion (Add Scheduled Noise)",
    "LatentHybridInverter": "Latent Hybrid Inverter (Qwen)",
    "ConditioningAddNoise": "Conditioning (Add Noise)",
}
