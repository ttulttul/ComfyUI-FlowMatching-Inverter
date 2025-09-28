import torch
import comfy.model_management
import comfy.utils

class QwenRectifiedFlowInverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent_image": ("LATENT",),
                "positive": ("CONDITIONING", ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "inversion_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "velocity_amplification": ("FLOAT", {
                    "default": 0.0, 
                    "min": -0.9, # -0.9 dampens velocity to 10% of original
                    "max": 2.0,  # 2.0 triples the velocity
                    "step": 0.05, 
                    "tooltip": "Amplifies (>0) or dampens (<0) the predicted velocity at each step. Deterministic effect."
                }),
                "internal_precision": (["bfloat16", "float16", "force_float32"], {"default": "bfloat16"}),
                "debug_print": (["disable", "enable"], {"default": "disable"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "invert"
    CATEGORY = "Qwen/Sampling"
        
    def invert(self, model, latent_image, positive, steps, inversion_strength, velocity_amplification, internal_precision, debug_print):
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

                # --- NEW: Velocity Amplification ---
                # This is a deterministic modification of the velocity vector.
                if velocity_amplification != 0.0:
                    predicted_velocity = predicted_velocity * (1.0 + velocity_amplification)
                # --- END NEW ---

                dt = t_next - t_current
                x_t = x_t + predicted_velocity * dt

                if torch.isnan(x_t).any():
                    print(f"!!! NaN detected at step {i+1}. Stopping inversion. !!!")
                    x_t.fill_(0) 
                    break
                
                pbar.update(1)
        
        out = latent_image.copy()
        out["samples"] = x_t.cpu()
        return (out, )

NODE_CLASS_MAPPINGS = { "QwenRectifiedFlowInverter": QwenRectifiedFlowInverter }
NODE_DISPLAY_NAME_MAPPINGS = { "QwenRectifiedFlowInverter": "Qwen Rectified Flow Inverter" }
