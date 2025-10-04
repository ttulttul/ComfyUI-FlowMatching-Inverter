# ComfyUI Flow Matching Inverter Nodes

Custom nodes for ComfyUI that target rectified-flow models (for example, Qwen). The nodes cover three core tasks:

- integrate the model's velocity field to produce a partially inverted latent,
- add noise using the same schedule as the sampler, and
- perform small latent-domain cleanups or conditioning perturbations.

Each node is available on its own, and the hybrid node chains them together for a single-drop workflow.

## Conditioning primer

ComfyUI passes prompt information around as a Python list of `[embedding, metadata]` pairs. The `embedding` tensor stores the per-token hidden states produced by the text encoder (shape ≈ `batch × tokens × features`, or `tokens × features` when the batch dimension is implicit). The accompanying `metadata` dictionary carries auxiliary fields—most notably `pooled_output`, a single vector that summarises the whole prompt (CLS token, sentence average, etc.).

Whenever you tweak conditioning with the nodes below you are directly manipulating those tensors. Splitting, blurring, or scaling happens along the token axis, while pooled outputs are updated in lockstep so downstream samplers still receive a coherent summary.

## Node overview

### Qwen Rectified Flow Inverter

Runs the rectified-flow ODE backwards to re-noise an encoded latent. Supports deterministic amplification, seeded stochastic perturbation, optional output normalisation, and shares the same interface as the sampler (`steps`, `conditioning`, etc.).

### Latent Gaussian Blur

Applies a Gaussian blur directly in latent space. `Spatial Only` blurs each channel independently; `Spatial and Channel` performs a joint 3D blur across channels for stronger smoothing.

### Latent Frequency Split

Separates a latent into low-pass structure and high-frequency detail bands using a Gaussian crossover so you can process each side differently before recombining.

### Add Latent Noise (Seeded)

Adds seeded Gaussian noise scaled by the input latent's standard deviation. Useful whenever a latent needs a controlled bump in noise without touching the schedule.

### Latent Perlin Fractal Noise

Generates smooth octave-based Perlin noise and adds it to the latent. Tune the base frequency, octaves, persistence, and lacunarity to decide whether you want broad undulations or fine stippling, and choose between shared or per-channel fields for subtle colour-channel offsets.

### Latent Swirl Noise

Warps the latent around one or more seeded vortices using `grid_sample`. Control the vortex count, maximum rotation (in radians), falloff radius, centre jitter, and blend strength to inject painterly whirlpools or gentle twisting motion into the underlying features.

### Conditioning (Add Noise)

Adds seeded Gaussian noise to the CLIP conditioning embeddings (and pooled output when present). Great for introducing gentle prompt variation without rewriting text.

Sensible values of `strength` range from 0.0 to 1.5. Above this range, the Qwen model's output tends to get silly, with Chinese characters overlaid on the image.

### Conditioning (Gaussian Blur)

Smooths the token embeddings with a Gaussian kernel along the prompt sequence, softening sharp emphasis changes while preserving the overall prompt content.

### Conditioning (Frequency Split)

Generates low- and high-frequency prompt embeddings so you can tame the overall narrative while separately shaping high-energy emphasis tokens.

Sensible values of `sigma` range from 0.1 to 1.3. The higher you raise `sigma`, the more of the conditioning information that will be shoved into the `low_pass` conditioning output. Values well above 1.0 are really equivalent to knocking out the high pass signal entirely. But strange and interesting results can be obtained by using a very high sigma in combination with a very high value of `gain` for the `high_pass` conditioning in the `Conditioning (Frequency Merge)` node.

### Conditioning (Frequency Merge)

Recombines low/high conditioning bands with adjustable gains so you can dial detail back in after sculpting each band independently. Boosting the low band reinforces the prompt's broad narrative and stability, while boosting the high band pushes sharp emphasis changes and punctuation to the forefront.

Feed the output of the `Conditioning (Frequency Split)` node into the input of this node and then play around with the gain knobs for interesting effects. Or peel off the high or low-pass conditioning and then add noise to one or the other, or combine the high pass conditionings of two different `CLIP Text Encode` outputs... the combinations are endless.

### Conditioning (Scale)

Multiplies the conditioning embeddings and pooled outputs by a user-defined factor so you can mute (0.0), keep (1.0), or amplify (>1.0) prompt influence without editing text.

### Forward Diffusion (Add Scheduled Noise)

Uses the model's noise schedule (via `KSampler`) to add the amount of noise that corresponds to a given progress value. This is the "anchor" latent that samplers expect when starting from a given denoise level.

### Latent Hybrid Inverter (Qwen)

Convenience node that calls the inverter and forward diffusion nodes internally, then blends their outputs using spherical interpolation. It keeps separate seeds for the creative and anchor paths and exposes the blend weight so you can bias towards either latent.

## Creative play

Image exploration rarely means chasing a single "correct" answer. These nodes are designed to be mixed and layered so you can steer a latent in different, often surprising, directions:

- Lead with `Latent Hybrid Inverter` to recover structure, then nudge that structure by adding `Conditioning (Add Noise)` at a subtle strength (≈0.05–0.15). The latent stays coherent while the prompt wanders just enough to spark new ideas.
- Follow the noise with `Conditioning (Gaussian Blur)` to soften abrupt emphasis changes or to merge multi-prompt blends into a single vibe. Blurring after noise often produces dreamlike, painterly shifts instead of chaotic drift.
- Use `Conditioning (Scale)` as a volume knob while iterating: dial the factor down to 0.3–0.5 when you want the image to respond mostly to the recovered latent, then crank it above 1.5 when the textual guidance should take the lead.
- Pair `Latent Gaussian Blur` with `Conditioning (Gaussian Blur)` for holistic smoothing—latent blur calms texture while conditioning blur calms prompt pacing. Reintroduce energy by sprinkling in `Add Latent Noise` or ramping the conditioning scale afterwards.
- Sculpt structured detail by layering `Latent Perlin Fractal Noise` (shared mode for broad waves, per-channel for colour shifts) before blending back with `Latent Swirl Noise`—increase `vortices` for multi-spiral motion, or keep it at 1 for a single focal swirl.
- Split a latent with `Latent Frequency Split`, sharpen or noise the high band while blurring the low band, then recombine via `Latent Mixer`/`Add` nodes to weave sharp detail onto soft composition.
- Carve prompts with `Conditioning (Frequency Split)` and return them through `Conditioning (Frequency Merge)`: keep low-band text steady for the story while remapping the high band through noise, blur, or scale to emphasize only certain fragments.
- Swap seeds between the inverter, forward diffusion, and noise nodes as you iterate. Small tweaks here can shift the interplay between latent detail and prompt guidance, rewarding playful experimentation.

Treat each slider as a brushstroke: push a setting until it breaks, back off to the sweet spot, and capture the happy accidents along the way.

## Example workflows

### Manual composition

1. Encode an image to a latent.
2. Feed the latent into `Qwen Rectified Flow Inverter` (pick strength/amplification/perturbation).
3. Feed the clean latent into `Forward Diffusion` with the same strength but a different seed.
4. Optionally clean either latent with `Latent Gaussian Blur` or `Add Latent Noise`.
5. Blend the creative and anchor latents (e.g. `Latent Slerp`) and sample with `denoise` equal to the chosen strength.

### Hybrid node quickstart

1. Encode an image to a latent.
2. Run the latent through `Latent Hybrid Inverter` with your preferred strength and blend factor.
3. Sample the result with `denoise` matching the passed strength.

## Installation

1. Clone or download this repository into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI to load the nodes (look under `Qwen/Sampling` and `Latent`).
