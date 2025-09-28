# ComfyUI Qwen Rectified Flow Tools

This custom node pack for ComfyUI provides a suite of advanced tools for performing "invert-and-resample" workflows with Rectified Flow models like Qwen-VL. It allows you to take an existing image, convert it to a noisy latent using a guided inversion process, and then use that latent as a starting point to generate creative, high-quality variations.

The pack provides both the individual building blocks for maximum flexibility and a final "convenience node" that encapsulates the optimal workflow for ease of use.

## Features

-   **Guided Latent Inversion:** Go beyond simple noise addition. Use the model's own understanding of the image (its velocity field) to intelligently re-noise a latent.
-   **Creative Controls:** Deterministically **amplify** the inversion process or stochastically **perturb** it with seeded noise to create unique variations.
-   **Stability Tools:** Includes specialized nodes to solve common artifacts in advanced workflows, such as VAE "waffle patterns" and sampler "brown-out" collapse.
-   **Modular and Combined Nodes:** Use the individual components to experiment, or use the final `Latent Hybrid Inverter` for a powerful, one-node solution.

## Nodes Included

### 1. Qwen Rectified Flow Inverter

This is the core node of the pack. It takes a clean latent and "re-noises" it by running the Rectified Flow ODE integration in reverse.

-   **Key Inputs:**
    -   `inversion_strength`: How far to re-noise the image (0.0 = no change, 1.0 = full noise).
    -   `velocity_amplification`: Deterministically speeds up or slows down the inversion.
    -   `velocity_perturb_strength`: Adds seeded random chaos to the inversion path for creative variations.
    -   `normalize_output`: **Crucial for stability.** Fixes the statistics of the output latent to prevent sampler errors.

### 2. Latent Gaussian Blur

A utility node to fix high-frequency artifacts in latents. Its primary purpose is to eliminate the "waffle pattern" that can appear in img2img workflows.

-   **Key Inputs:**
    -   `sigma`: The strength of the blur. A small value (0.5-1.5) is usually enough.
    -   `blur_mode`: Choose between `Spatial Only` (safe, recommended) or the experimental `Spatial and Channel` for artistic effects.

### 3. Add Latent Noise (Seeded)

A simple but powerful utility to add a configurable amount of seeded Gaussian noise to any latent. Useful for re-introducing "perfect" noise to a latent that has an unnatural distribution.

-   **Key Inputs:**
    -   `strength`: How much noise to add, relative to the latent's existing signal strength.
    -   `seed`: Makes the random noise pattern repeatable.

### 4. Forward Diffusion (Add Scheduled Noise)

This node performs the "natural" noising process that diffusion models are trained on. It takes a clean latent and adds the mathematically "perfect" amount of noise for a given step in the schedule. This creates a stable "anchor" latent.

-   **Key Inputs:**
    -   `noise_strength`: The target noise level (0.0 to 1.0). Should match your KSampler's `denoise` value.

### 5. Latent Hybrid Inverter (Qwen)

The final convenience node that combines the best practices discovered into a single, powerful tool. It internally performs the creative inversion, creates a stable anchor latent via forward diffusion, and blends them together to produce an ideal starting latent for resampling.

-   **Key Inputs:**
    -   `strength`: The target noise level for both passes.
    -   `blend_factor`: Controls the mix between the creative inverter latent (1.0) and the stable anchor latent (0.0). A 50/50 blend (`0.5`) is recommended.
    -   Contains all the creative controls from the inverter and separate seeds for each internal process.

## Workflows

### Basic Invert and Resample (Img2Img)

This workflow demonstrates the basic principle but may be prone to artifacts at high denoise strengths.

1.  **VAE Encode** your source image.
2.  Connect the latent to the **Qwen Rectified Flow Inverter**.
    -   Set `inversion_strength` to `0.8`.
3.  Connect the output to a **KSampler**.
    -   **Crucially, set `denoise` to match the `inversion_strength` (`0.8`).**
4.  **VAE Decode** the result.

### Advanced: The "Anchor" Workflow (Recommended for High Quality)

This is the optimal workflow that solves both the "waffle pattern" and "brown-out" artifacts, allowing for high-strength, high-quality variations.

1.  **VAE Encode** the source image.
2.  **Create the "Creative" Latent:**
    -   Feed the clean latent into the **Qwen Rectified Flow Inverter**. Set your desired `inversion_strength`, `amplification`, `perturbation`, etc.
    -   (Optional but Recommended) Feed the result into a **Latent Gaussian Blur** with a low `sigma` (e.g., 1.0) to remove VAE artifacts.
3.  **Create the "Anchor" Latent:**
    -   Feed the original clean latent from the VAE Encode into the **Forward Diffusion (Add Scheduled Noise)** node.
    -   Set its `noise_strength` to *exactly match* the `inversion_strength` from the inverter. Use a different seed.
4.  **Blend the Latents:**
    -   Use a `Latent Blend` or `Latent Slerp` node to mix the "Creative" and "Anchor" latents. A blend factor of `0.5` is a great starting point.
5.  **Sample from the Blend:**
    -   Feed the blended latent into a **KSampler** (or **KSampler Advanced** with `add_noise` disabled).
    -   Set the KSampler's `denoise` value to *exactly match* the `strength` used in the previous steps.
6.  **VAE Decode** the final image.

![image](https://user-images.githubusercontent.com/12345/67890.png)*(Note: A diagram of the advanced workflow would be ideal here)*

### Ultimate Convenience: The Hybrid Inverter Workflow

This workflow uses the single convenience node to achieve the same result as the advanced workflow with fewer nodes.

1.  **VAE Encode** the source image.
2.  Connect the latent to the **Latent Hybrid Inverter**.
    -   Set the desired `strength` (e.g., 0.75).
    -   Set the `blend_factor` (e.g., 0.5).
    -   Configure seeds and creative parameters.
3.  Connect the output to a **KSampler**.
    -   Set the `denoise` to **match the `strength`** from the hybrid inverter.
4.  **VAE Decode** the result.

## Installation

1.  Clone or download this repository.
2.  Place the entire folder (e.g., `ComfyUI-Qwen-Inverter-Pack`) into the `ComfyUI/custom_nodes/` directory.
3.  Restart ComfyUI. The nodes will be available under the `Qwen/Sampling` and `Latent` categories.
