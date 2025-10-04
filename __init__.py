import math

import comfy.model_management
import comfy.utils

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a, b, t):
    return a + t * (b - a)


def _generate_permutation(seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(256, generator=generator)
    return torch.cat([perm, perm]).to(device)


def _coordinate_grid(size, device, dtype):
    axes = [torch.linspace(0.0, 1.0, steps=s, device=device, dtype=dtype) for s in size]
    return torch.meshgrid(*axes, indexing="ij")


def _perlin_2d(x, y, perm, dtype):
    gradients = torch.tensor(
        [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]],
        dtype=dtype,
        device=x.device,
    )

    x0 = torch.floor(x)
    y0 = torch.floor(y)

    xi = torch.remainder(x0.to(torch.int64), 256)
    yi = torch.remainder(y0.to(torch.int64), 256)

    xf = x - x0
    yf = y - y0

    xi1 = torch.remainder(xi + 1, 256)
    yi1 = torch.remainder(yi + 1, 256)

    perm_xi = perm[xi]
    perm_xi1 = perm[xi1]

    aa = perm[perm_xi + yi]
    ab = perm[perm_xi + yi1]
    ba = perm[perm_xi1 + yi]
    bb = perm[perm_xi1 + yi1]

    grad_count = gradients.shape[0]

    g_aa = gradients[torch.remainder(aa, grad_count)]
    g_ab = gradients[torch.remainder(ab, grad_count)]
    g_ba = gradients[torch.remainder(ba, grad_count)]
    g_bb = gradients[torch.remainder(bb, grad_count)]

    xf_1 = xf - 1.0
    yf_1 = yf - 1.0

    dot_aa = g_aa[..., 0] * xf + g_aa[..., 1] * yf
    dot_ba = g_ba[..., 0] * xf_1 + g_ba[..., 1] * yf
    dot_ab = g_ab[..., 0] * xf + g_ab[..., 1] * yf_1
    dot_bb = g_bb[..., 0] * xf_1 + g_bb[..., 1] * yf_1

    u = _fade(xf)
    v = _fade(yf)

    x1 = _lerp(dot_aa, dot_ba, u)
    x2 = _lerp(dot_ab, dot_bb, u)
    return _lerp(x1, x2, v)


def _perlin_3d(x, y, z, perm, dtype):
    gradients = torch.tensor(
        [
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1],
        ],
        dtype=dtype,
        device=x.device,
    )

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    z0 = torch.floor(z)

    xi = torch.remainder(x0.to(torch.int64), 256)
    yi = torch.remainder(y0.to(torch.int64), 256)
    zi = torch.remainder(z0.to(torch.int64), 256)

    xf = x - x0
    yf = y - y0
    zf = z - z0

    xi1 = torch.remainder(xi + 1, 256)
    yi1 = torch.remainder(yi + 1, 256)
    zi1 = torch.remainder(zi + 1, 256)

    perm_xi = perm[xi]
    perm_xi1 = perm[xi1]

    perm_xy = perm[perm_xi + yi]
    perm_xy1 = perm[perm_xi + yi1]
    perm_x1y = perm[perm_xi1 + yi]
    perm_x1y1 = perm[perm_xi1 + yi1]

    aaa = perm[perm_xy + zi]
    aab = perm[perm_xy + zi1]
    aba = perm[perm_xy1 + zi]
    abb = perm[perm_xy1 + zi1]
    baa = perm[perm_x1y + zi]
    bab = perm[perm_x1y + zi1]
    bba = perm[perm_x1y1 + zi]
    bbb = perm[perm_x1y1 + zi1]

    grad_count = gradients.shape[0]

    g_aaa = gradients[torch.remainder(aaa, grad_count)]
    g_aab = gradients[torch.remainder(aab, grad_count)]
    g_aba = gradients[torch.remainder(aba, grad_count)]
    g_abb = gradients[torch.remainder(abb, grad_count)]
    g_baa = gradients[torch.remainder(baa, grad_count)]
    g_bab = gradients[torch.remainder(bab, grad_count)]
    g_bba = gradients[torch.remainder(bba, grad_count)]
    g_bbb = gradients[torch.remainder(bbb, grad_count)]

    xf_1 = xf - 1.0
    yf_1 = yf - 1.0
    zf_1 = zf - 1.0

    dot_aaa = g_aaa[..., 0] * xf + g_aaa[..., 1] * yf + g_aaa[..., 2] * zf
    dot_baa = g_baa[..., 0] * xf_1 + g_baa[..., 1] * yf + g_baa[..., 2] * zf
    dot_aba = g_aba[..., 0] * xf + g_aba[..., 1] * yf_1 + g_aba[..., 2] * zf
    dot_bba = g_bba[..., 0] * xf_1 + g_bba[..., 1] * yf_1 + g_bba[..., 2] * zf
    dot_aab = g_aab[..., 0] * xf + g_aab[..., 1] * yf + g_aab[..., 2] * zf_1
    dot_bab = g_bab[..., 0] * xf_1 + g_bab[..., 1] * yf + g_bab[..., 2] * zf_1
    dot_abb = g_abb[..., 0] * xf + g_abb[..., 1] * yf_1 + g_abb[..., 2] * zf_1
    dot_bbb = g_bbb[..., 0] * xf_1 + g_bbb[..., 1] * yf_1 + g_bbb[..., 2] * zf_1

    u = _fade(xf)
    v = _fade(yf)
    w = _fade(zf)

    x1 = _lerp(dot_aaa, dot_baa, u)
    x2 = _lerp(dot_aba, dot_bba, u)
    y1 = _lerp(x1, x2, v)

    x3 = _lerp(dot_aab, dot_bab, u)
    x4 = _lerp(dot_abb, dot_bbb, u)
    y2 = _lerp(x3, x4, v)

    return _lerp(y1, y2, w)


def _fractal_perlin(size, seed, frequency, octaves, persistence, lacunarity, device):
    if len(size) not in (2, 3):
        raise ValueError("Perlin noise supports 2D or 3D shapes")

    noise_dtype = torch.float32
    coords = _coordinate_grid(size, device, noise_dtype)

    total = torch.zeros(size, dtype=noise_dtype, device=device)
    amplitude = 1.0
    max_amplitude = 0.0
    freq = frequency

    for octave in range(octaves):
        perm = _generate_permutation(seed + octave, device)

        scaled_coords = [coord * freq for coord in coords]

        if len(size) == 2:
            octave_noise = _perlin_2d(scaled_coords[1], scaled_coords[0], perm, noise_dtype)
        else:
            octave_noise = _perlin_3d(scaled_coords[2], scaled_coords[1], scaled_coords[0], perm, noise_dtype)

        total = total + octave_noise * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        freq *= lacunarity

    if max_amplitude > 0:
        total = total / max_amplitude

    return total



def _simplex_noise_2d(x, y, perm, dtype):
    F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    G2 = (3.0 - math.sqrt(3.0)) / 6.0

    s = (x + y) * F2
    i = torch.floor(x + s)
    j = torch.floor(y + s)

    t = (i + j) * G2
    X0 = i - t
    Y0 = j - t
    x0 = x - X0
    y0 = y - Y0

    cond = x0 > y0
    i1 = torch.where(cond, torch.ones_like(x0), torch.zeros_like(x0))
    j1 = torch.where(cond, torch.zeros_like(y0), torch.ones_like(y0))

    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2

    ii = torch.remainder(i.to(torch.int64), 256)
    jj = torch.remainder(j.to(torch.int64), 256)

    i1_int = i1.to(torch.int64)
    j1_int = j1.to(torch.int64)

    gi0 = perm[ii + perm[jj]] % 12
    gi1 = perm[ii + i1_int + perm[jj + j1_int]] % 12
    gi2 = perm[ii + 1 + perm[jj + 1]] % 12

    gradients = torch.tensor(
        [
            [1, 1],
            [-1, 1],
            [1, -1],
            [-1, -1],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1, 1] / math.sqrt(2.0),
            [-1, 1] / math.sqrt(2.0),
            [1, -1] / math.sqrt(2.0),
            [-1, -1] / math.sqrt(2.0),
        ],
        dtype=dtype,
        device=x.device,
    )

    def contribution(t_comp, gx, gy, x_comp, y_comp):
        mask = t_comp > 0
        t_term = torch.zeros_like(t_comp)
        if mask.any():
            t_square = t_comp[mask] * t_comp[mask]
            t_four = t_square * t_square
            dot = gx[mask] * x_comp[mask] + gy[mask] * y_comp[mask]
            t_term[mask] = t_four * dot
        return t_term

    g0 = gradients[gi0]
    g1 = gradients[gi1]
    g2 = gradients[gi2]

    t0 = 0.5 - x0 * x0 - y0 * y0
    t1 = 0.5 - x1 * x1 - y1 * y1
    t2 = 0.5 - x2 * x2 - y2 * y2

    n0 = contribution(t0, g0[..., 0], g0[..., 1], x0, y0)
    n1 = contribution(t1, g1[..., 0], g1[..., 1], x1, y1)
    n2 = contribution(t2, g2[..., 0], g2[..., 1], x2, y2)

    return 70.0 * (n0 + n1 + n2)


def _fractal_simplex(size, seed, frequency, octaves, persistence, lacunarity, device):
    if len(size) != 2:
        raise ValueError("Simplex noise is implemented for 2D shapes")

    dtype = torch.float32
    coords = _coordinate_grid(size, device, dtype)

    total = torch.zeros(size, dtype=dtype, device=device)
    amplitude = 1.0
    max_amplitude = 0.0
    freq = frequency

    for octave in range(octaves):
        perm = _generate_permutation(seed + octave, device)
        x = coords[1] * freq
        y = coords[0] * freq
        octave_noise = _simplex_noise_2d(x, y, perm, dtype)
        total = total + octave_noise * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        freq *= lacunarity

    if max_amplitude > 0:
        total = total / max_amplitude

    return total


def _worley_noise(size, seed, feature_points, metric, jitter, device):
    if len(size) != 2:
        raise ValueError("Worley noise expects a 2D shape")

    height, width = size
    dtype = torch.float32
    cpu_gen = torch.Generator(device="cpu").manual_seed(seed)
    points = torch.rand((feature_points, 2), generator=cpu_gen, dtype=dtype)

    if jitter > 0.0:
        jitter_gen = torch.rand((feature_points, 2), generator=cpu_gen, dtype=dtype) - 0.5
        points = torch.clamp(points + jitter_gen * jitter, 0.0, 1.0)

    points = points.to(device=device)

    ys = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
    grid_y = ys.view(height, 1, 1)
    grid_x = xs.view(1, width, 1)

    diff_y = grid_y - points[:, 1]
    diff_x = grid_x - points[:, 0]

    diff = torch.stack((diff_x, diff_y), dim=-1)

    if metric == "manhattan":
        distances = diff.abs().sum(dim=-1)
    elif metric == "chebyshev":
        distances = diff.abs().max(dim=-1).values
    else:
        distances = torch.sqrt((diff * diff).sum(dim=-1))

    min_dist = distances.min(dim=-1).values
    max_dist = torch.max(min_dist)
    if max_dist > 0:
        min_dist = min_dist / max_dist

    return 1.0 - min_dist


def _fractal_worley(size, seed, base_points, octaves, persistence, lacunarity, metric, jitter, device):
    dtype = torch.float32
    total = torch.zeros(size, dtype=dtype, device=device)
    amplitude = 1.0
    max_amplitude = 0.0
    points = base_points

    for octave in range(octaves):
        octave_points = max(1, int(round(points)))
        noise = _worley_noise(size, seed + octave, octave_points, metric, jitter, device)
        total = total + noise * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        points *= lacunarity

    if max_amplitude > 0:
        total = total / max_amplitude

    return total * 2.0 - 1.0


def _gray_scott_pattern(size, seed, steps, feed, kill, diff_u, diff_v, dt, device):
    if len(size) != 2:
        raise ValueError("Gray-Scott pattern expects a 2D shape")

    height, width = size
    dtype = torch.float32

    U = torch.ones((1, 1, height, width), device=device, dtype=dtype)
    V = torch.zeros((1, 1, height, width), device=device, dtype=dtype)

    cpu_gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand((height, width), generator=cpu_gen, dtype=dtype)
    noise = noise.to(device=device)

    U = U - 0.5 * noise.unsqueeze(0).unsqueeze(0)
    V = V + 0.5 * noise.unsqueeze(0).unsqueeze(0)

    square = max(2, min(height, width) // 10)
    start_y = height // 2 - square // 2
    start_x = width // 2 - square // 2
    V[:, :, start_y:start_y + square, start_x:start_x + square] = 1.0
    U[:, :, start_y:start_y + square, start_x:start_x + square] = 0.0

    lap_kernel = torch.tensor(
        [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)

    for _ in range(steps):
        U_pad = torch.nn.functional.pad(U, (1, 1, 1, 1), mode="reflect")
        V_pad = torch.nn.functional.pad(V, (1, 1, 1, 1), mode="reflect")
        lap_u = torch.nn.functional.conv2d(U_pad, lap_kernel)
        lap_v = torch.nn.functional.conv2d(V_pad, lap_kernel)

        reaction = U * V * V
        U = U + (diff_u * lap_u - reaction + feed * (1.0 - U)) * dt
        V = V + (diff_v * lap_v + reaction - (feed + kill) * V) * dt

        U = torch.clamp(U, 0.0, 1.0)
        V = torch.clamp(V, 0.0, 1.0)

    pattern = V.squeeze(0).squeeze(0)
    return _normalize_noise_tensor(pattern)


def _apply_2d_noise(latent_tensor, seed, strength, channel_mode, temporal_mode, generator):
    if strength == 0.0:
        return latent_tensor

    with torch.no_grad():
        device = latent_tensor.device
        is_video = latent_tensor.dim() == 5
        if is_video:
            batch, channels, frames, height, width = latent_tensor.shape
        else:
            batch, channels, height, width = latent_tensor.shape
            frames = 1

        output = latent_tensor.clone()

        for batch_index in range(batch):
            sample = latent_tensor[batch_index]
            sample_std = torch.std(sample.float())
            scale = sample_std.item() * strength if sample_std > 1e-6 else strength

            if channel_mode == "shared":
                if is_video:
                    if temporal_mode == "locked":
                        base_noise = generator((height, width), seed + batch_index)
                        base_noise = _normalize_noise_tensor(base_noise)
                        noise = base_noise.unsqueeze(0).unsqueeze(0).expand(channels, frames, height, width).clone()
                    else:
                        noise = torch.zeros((channels, frames, height, width), dtype=torch.float32, device=device)
                        for frame_index in range(frames):
                            frame_seed = seed + batch_index + frame_index * 131
                            frame_noise = generator((height, width), frame_seed)
                            frame_plane = _normalize_noise_tensor(frame_noise).unsqueeze(0).expand(channels, height, width)
                            noise[:, frame_index] = frame_plane.clone()
                else:
                    base_noise = generator((height, width), seed + batch_index)
                    base_noise = _normalize_noise_tensor(base_noise)
                    noise = base_noise.unsqueeze(0).expand(channels, height, width).clone()
            else:
                if is_video:
                    noise = torch.zeros((channels, frames, height, width), dtype=torch.float32, device=device)
                    for channel_index in range(channels):
                        channel_seed = seed + batch_index * 997 + channel_index * 1013
                        if temporal_mode == "locked":
                            channel_noise = generator((height, width), channel_seed)
                            normalized = _normalize_noise_tensor(channel_noise)
                            noise[channel_index] = normalized.unsqueeze(0).expand(frames, height, width).clone()
                        else:
                            for frame_index in range(frames):
                                frame_seed = channel_seed + frame_index * 131
                                frame_noise = generator((height, width), frame_seed)
                                noise[channel_index, frame_index] = _normalize_noise_tensor(frame_noise)
                else:
                    noise = torch.zeros((channels, height, width), dtype=torch.float32, device=device)
                    for channel_index in range(channels):
                        channel_seed = seed + batch_index * 997 + channel_index * 1013
                        channel_noise = generator((height, width), channel_seed)
                        noise[channel_index] = _normalize_noise_tensor(channel_noise)

            scaled_noise = (noise * scale).to(sample.dtype)
            output[batch_index] = sample + scaled_noise

    return output

def _normalize_noise_tensor(noise):
    mean = torch.mean(noise)
    std = torch.std(noise)
    if std > 1e-6:
        return (noise - mean) / std
    return noise - mean


def _normalized_meshgrid(height, width, device, dtype):
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    return torch.meshgrid(ys, xs, indexing="ij")


def _swirl_grid(base_y, base_x, vortices):
    total_dx = torch.zeros_like(base_x)
    total_dy = torch.zeros_like(base_y)

    for center_x, center_y, strength, radius, direction in vortices:
        radius = max(radius, 1e-3)
        dx = base_x - center_x
        dy = base_y - center_y
        r = torch.sqrt(dx * dx + dy * dy)
        falloff = torch.exp(-torch.square(r / radius))
        angle = direction * strength * falloff
        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)
        rotated_dx = dx * cos_a - dy * sin_a
        rotated_dy = dx * sin_a + dy * cos_a
        total_dx = total_dx + (rotated_dx - dx)
        total_dy = total_dy + (rotated_dy - dy)

    x_new = base_x + total_dx
    y_new = base_y + total_dy
    grid = torch.stack((x_new, y_new), dim=-1)
    return torch.clamp(grid, -1.0, 1.0)


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
                "latent": ("LATENT", {"tooltip": "Latent tensor to blur."}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Standard deviation for the Gaussian kernel."}),
                "blur_mode": (["Spatial Only", "Spatial and Channel"], {"default": "Spatial Only", "tooltip": "Choose whether to blur across spatial dimensions only or include channels."}),
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


class LatentFrequencySplit:
    """Splits a latent into low- and high-frequency bands via Gaussian smoothing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to decompose into low/high frequency bands."}),
                "sigma": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Radius of the Gaussian used for the low-pass. Higher values move detail into the high band."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("low_pass", "high_pass")
    FUNCTION = "split"
    CATEGORY = "Latent/Filter"

    def split(self, latent, sigma):
        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].clone().to(device)

        if sigma <= 0.0:
            low_samples = latent_tensor
            high_samples = torch.zeros_like(latent_tensor)
        else:
            low_samples = self._gaussian_blur(latent_tensor, sigma)
            high_samples = latent_tensor - low_samples

        low_latent = latent.copy()
        low_latent["samples"] = low_samples.cpu()

        high_latent = latent.copy()
        high_latent["samples"] = high_samples.cpu()

        return (low_latent, high_latent)

    @staticmethod
    def _gaussian_blur(latent_tensor, sigma):
        tensor = latent_tensor
        is_5d = tensor.dim() == 5

        if is_5d:
            b, c, t, h, w = tensor.shape
            tensor_4d = tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        else:
            tensor_4d = tensor

        kernel_size = int(sigma * 6) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred_4d = TF.gaussian_blur(tensor_4d, kernel_size=kernel_size, sigma=sigma)

        if is_5d:
            blurred = blurred_4d.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        else:
            blurred = blurred_4d

        return blurred

class LatentAddNoise:
    """
    Adds a configurable amount of seeded random noise to a latent tensor.
    The strength is relative to the standard deviation of the input latent.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to receive additional Gaussian noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for generating repeatable noise."}),
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


class LatentPerlinFractalNoise:
    """Adds smooth fractal Perlin noise to latent tensors for structured variation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent that will be perturbed with fractal Perlin noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed controlling the procedural noise pattern."}),
                "frequency": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.01,
                    "max": 64.0,
                    "step": 0.01,
                    "tooltip": "Base lattice frequency. Higher values produce finer details."
                }),
                "octaves": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 12,
                    "tooltip": "Number of noise layers to accumulate."
                }),
                "persistence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Amplitude multiplier between octaves."
                }),
                "lacunarity": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 6.0,
                    "step": 0.1,
                    "tooltip": "Frequency multiplier between octaves."
                }),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Scales the normalized noise relative to the latent's standard deviation."
                }),
                "channel_mode": (["shared", "per_channel"], {
                    "default": "shared",
                    "tooltip": "Shared noise across channels or unique noise per channel."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_perlin_noise"
    CATEGORY = "Latent/Noise"

    def add_perlin_noise(self, latent, seed, frequency, octaves, persistence, lacunarity, strength, channel_mode):
        if strength == 0.0:
            return (latent,)

        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].clone().to(device)
        is_video = latent_tensor.dim() == 5

        if is_video:
            dims = (latent_tensor.shape[2], latent_tensor.shape[3], latent_tensor.shape[4])
        else:
            dims = (latent_tensor.shape[-2], latent_tensor.shape[-1])

        batch = latent_tensor.shape[0]
        channels = latent_tensor.shape[1]

        output = latent_tensor.clone()

        for batch_index in range(batch):
            sample = latent_tensor[batch_index]
            sample_std = torch.std(sample.float())

            if channel_mode == "shared":
                noise = _fractal_perlin(dims, seed + batch_index, frequency, octaves, persistence, lacunarity, device)
                noise = _normalize_noise_tensor(noise)

                if is_video:
                    noise = noise.unsqueeze(0).expand(channels, dims[0], dims[1], dims[2])
                else:
                    noise = noise.unsqueeze(0).expand(channels, dims[-2], dims[-1])
            else:
                if is_video:
                    noise = torch.zeros((channels, dims[0], dims[1], dims[2]), dtype=torch.float32, device=device)
                else:
                    noise = torch.zeros((channels, dims[-2], dims[-1]), dtype=torch.float32, device=device)

                for channel_index in range(channels):
                    channel_seed = seed + batch_index * 997 + channel_index * 1013
                    channel_noise = _fractal_perlin(dims, channel_seed, frequency, octaves, persistence, lacunarity, device)
                    noise[channel_index] = _normalize_noise_tensor(channel_noise)

            scale = sample_std * strength
            scaled_noise = (noise * scale).to(sample.dtype)
            output[batch_index] = sample + scaled_noise

        out = latent.copy()
        out["samples"] = output.cpu()

        return (out,)


class LatentSimplexNoise:
    """Applies layered simplex noise for organic latent perturbations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to perturb with simplex noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed controlling the simplex lattice offsets."}),
                "frequency": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 64.0, "step": 0.01, "tooltip": "Base lattice frequency for the simplex grid."}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 12, "tooltip": "Number of simplex layers to accumulate."}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amplitude multiplier applied between octaves."}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 6.0, "step": 0.1, "tooltip": "Frequency multiplier applied between octaves."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized simplex noise relative to the latent's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Reuse a single noise field for all channels or reseed per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reruns the same pattern per frame; animated reseeds each frame."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_simplex_noise"
    CATEGORY = "Latent/Noise"

    def add_simplex_noise(self, latent, seed, frequency, octaves, persistence, lacunarity, strength, channel_mode, temporal_mode):
        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].clone().to(device)

        generator = lambda size, noise_seed: _fractal_simplex(size, noise_seed, frequency, octaves, persistence, lacunarity, device)
        output = _apply_2d_noise(latent_tensor, seed, strength, channel_mode, temporal_mode, generator)

        out = latent.copy()
        out["samples"] = output.cpu()
        return (out,)


class LatentWorleyNoise:
    """Generates cellular Worley noise for cracked or biological textures."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to perturb with Worley (cellular) noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the feature point distribution."}),
                "feature_points": ("INT", {"default": 16, "min": 1, "max": 4096, "tooltip": "Base number of feature points scattered across the plane."}),
                "octaves": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "Number of cellular layers to accumulate."}),
                "persistence": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amplitude multiplier between Worley octaves."}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 6.0, "step": 0.1, "tooltip": "Multiplier for the feature point count between octaves."}),
                "distance_metric": (["euclidean", "manhattan", "chebyshev"], {"default": "euclidean", "tooltip": "Distance metric used when measuring feature proximity."}),
                "jitter": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How far feature points can drift inside each cell."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized Worley noise relative to the latent's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Shared noise for all channels or reseeded per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reuses a single noise pattern per frame; animated reseeds each frame."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_worley_noise"
    CATEGORY = "Latent/Noise"

    def add_worley_noise(self, latent, seed, feature_points, octaves, persistence, lacunarity, distance_metric, jitter, strength, channel_mode, temporal_mode):
        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].clone().to(device)

        base_points = float(max(1, feature_points))
        generator = lambda size, noise_seed: _fractal_worley(size, noise_seed, base_points, octaves, persistence, lacunarity, distance_metric, jitter, device)
        output = _apply_2d_noise(latent_tensor, seed, strength, channel_mode, temporal_mode, generator)

        out = latent.copy()
        out["samples"] = output.cpu()
        return (out,)


class LatentReactionDiffusion:
    """Runs a Gray-Scott reaction-diffusion simulation and injects the resulting pattern."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent that will receive reaction-diffusion patterns."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the initial chemical concentrations."}),
                "iterations": ("INT", {"default": 200, "min": 1, "max": 2000, "tooltip": "Number of Gray-Scott simulation steps."}),
                "feed_rate": ("FLOAT", {"default": 0.036, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Feed rate (F) controlling how quickly chemical U is replenished."}),
                "kill_rate": ("FLOAT", {"default": 0.065, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Kill rate (K) regulating removal of chemical V."}),
                "diffusion_u": ("FLOAT", {"default": 0.16, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Diffusion rate for chemical U."}),
                "diffusion_v": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Diffusion rate for chemical V."}),
                "time_step": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01, "tooltip": "Simulation time step used during integration."}),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized pattern relative to the latent's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Reuse one simulation for all channels or rerun per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reuses the same pattern for every frame; animated reruns the simulation per frame."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_reaction_diffusion"
    CATEGORY = "Latent/Noise"

    def add_reaction_diffusion(self, latent, seed, iterations, feed_rate, kill_rate, diffusion_u, diffusion_v, time_step, strength, channel_mode, temporal_mode):
        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].clone().to(device)

        generator = lambda size, noise_seed: _gray_scott_pattern(size, noise_seed, iterations, feed_rate, kill_rate, diffusion_u, diffusion_v, time_step, device)
        output = _apply_2d_noise(latent_tensor, seed, strength, channel_mode, temporal_mode, generator)

        out = latent.copy()
        out["samples"] = output.cpu()
        return (out,)


class LatentFractalBrownianMotion:
    """Builds fractal Brownian motion from a selectable base noise and injects it into the latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to enrich with fractal Brownian motion."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the base noise generator."}),
                "base_noise": (["simplex", "perlin", "worley"], {"default": "simplex", "tooltip": "Noise primitive accumulated by the fBm stack."}),
                "frequency": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 64.0, "step": 0.01, "tooltip": "Fundamental frequency for simplex/perlin bases (acts as a multiplier for Worley)."}),
                "feature_points": ("INT", {"default": 16, "min": 1, "max": 4096, "tooltip": "Base feature point count (used when base noise is Worley)."}),
                "octaves": ("INT", {"default": 5, "min": 1, "max": 12, "tooltip": "Number of fBm layers to accumulate."}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amplitude multiplier between fBm layers."}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 6.0, "step": 0.1, "tooltip": "Frequency multiplier between fBm layers."}),
                "distance_metric": (["euclidean", "manhattan", "chebyshev"], {"default": "euclidean", "tooltip": "Distance metric used when the base noise is Worley."}),
                "jitter": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Feature jitter amount for Worley base noise."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Scale of the normalized fBm field relative to the latent's standard deviation."}),
                "channel_mode": (["shared", "per_channel"], {"default": "shared", "tooltip": "Shared fBm field per sample or reseeded per channel."}),
                "temporal_mode": (["locked", "animated"], {"default": "locked", "tooltip": "locked reuses the same fBm per frame; animated reseeds each frame."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_fbm_noise"
    CATEGORY = "Latent/Noise"

    def add_fbm_noise(self, latent, seed, base_noise, frequency, feature_points, octaves, persistence, lacunarity, distance_metric, jitter, strength, channel_mode, temporal_mode):
        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].clone().to(device)

        def generator(size, noise_seed):
            if base_noise == "perlin":
                return _fractal_perlin(size, noise_seed, frequency, octaves, persistence, lacunarity, device)
            if base_noise == "worley":
                points = float(max(1, feature_points)) * max(frequency, 0.01)
                return _fractal_worley(size, noise_seed, points, octaves, persistence, lacunarity, distance_metric, jitter, device)
            return _fractal_simplex(size, noise_seed, frequency, octaves, persistence, lacunarity, device)

        output = _apply_2d_noise(latent_tensor, seed, strength, channel_mode, temporal_mode, generator)

        out = latent.copy()
        out["samples"] = output.cpu()
        return (out,)


class LatentSwirlNoise:
    """Swirls latent pixels around randomized centers for vortex-like perturbations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent to deform with vortex-style warps."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for vortex placement and direction randomness."}),
                "vortices": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Number of independent vortex centres to spawn per latent."
                }),
                "channel_mode": (["global", "per_channel"], {
                    "default": "global",
                    "tooltip": "Use a shared swirl grid for all channels or generate unique grids per affected channel."
                }),
                "channel_fraction": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Fraction of channels to swirl. The subset is randomly selected per sample."
                }),
                "strength": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 6.28,
                    "step": 0.01,
                    "tooltip": "Peak swirl rotation in radians near the vortex center."
                }),
                "radius": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.05,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Normalized radius controlling how far the vortex influence extends."
                }),
                "center_spread": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How far the vortex origin drifts from the latent center."
                }),
                "direction_bias": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Bias toward counter-clockwise (1) or clockwise (-1) swirl."
                }),
                "mix": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend between original latent (0) and fully swirled result (1)."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_swirl_noise"
    CATEGORY = "Latent/Noise"

    def add_swirl_noise(self, latent, seed, vortices, channel_mode, channel_fraction,
                        strength, radius, center_spread, direction_bias, mix):
        if strength == 0.0 or mix == 0.0:
            return (latent,)

        device = comfy.model_management.get_torch_device()
        latent_tensor = latent["samples"].clone().to(device)
        is_video = latent_tensor.dim() == 5

        if is_video:
            batch, channels, frames, height, width = latent_tensor.shape
            working = latent_tensor.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        else:
            batch, channels, height, width = latent_tensor.shape
            frames = 1
            working = latent_tensor

        base_dtype = torch.float32
        base_y, base_x = _normalized_meshgrid(height, width, device, base_dtype)
        result = torch.empty_like(working)

        center_spread = max(0.0, min(center_spread, 1.0))
        mix = max(0.0, min(mix, 1.0))
        bias_threshold = max(0.0, min(1.0, (direction_bias + 1.0) * 0.5))
        vortex_count = max(1, int(vortices))
        channel_fraction = max(0.0, min(1.0, channel_fraction))

        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)

        total = working.shape[0]

        for idx in range(total):
            if channel_fraction <= 0.0:
                result[idx] = working[idx]
                continue

            channel_count = working.shape[1]
            selected_count = min(
                channel_count,
                max(1, math.ceil(channel_count * channel_fraction))
            )

            channel_indices = torch.randperm(channel_count, generator=cpu_generator)[:selected_count]

            sample = working[idx:idx + 1]
            sample_result = sample.clone()

            if channel_mode == "global":
                rand_vals = torch.rand((vortex_count, 3), generator=cpu_generator)
                vortex_params = []

                for vortex_idx in range(vortex_count):
                    offsets = (rand_vals[vortex_idx, :2] * 2.0 - 1.0) * center_spread
                    center_y = offsets[0].item()
                    center_x = offsets[1].item()
                    direction = 1.0 if rand_vals[vortex_idx, 2].item() < bias_threshold else -1.0
                    vortex_params.append((center_x, center_y, float(strength), float(radius), direction))

                grid = _swirl_grid(base_y, base_x, vortex_params).unsqueeze(0)
                warped = F.grid_sample(
                    sample.to(base_dtype),
                    grid,
                    mode="bilinear",
                    padding_mode="reflection",
                    align_corners=True,
                ).to(sample.dtype)

                if mix >= 1.0:
                    sample_result[:, channel_indices] = warped[:, channel_indices]
                else:
                    delta = warped[:, channel_indices] - sample[:, channel_indices]
                    sample_result[:, channel_indices] = sample[:, channel_indices] + delta * mix

            else:  # per_channel
                for channel_idx in channel_indices.tolist():
                    rand_vals = torch.rand((vortex_count, 3), generator=cpu_generator)
                    vortex_params = []

                    for vortex_idx in range(vortex_count):
                        offsets = (rand_vals[vortex_idx, :2] * 2.0 - 1.0) * center_spread
                        center_y = offsets[0].item()
                        center_x = offsets[1].item()
                        direction = 1.0 if rand_vals[vortex_idx, 2].item() < bias_threshold else -1.0
                        vortex_params.append((center_x, center_y, float(strength), float(radius), direction))

                    grid = _swirl_grid(base_y, base_x, vortex_params).unsqueeze(0)

                    channel_slice = sample[:, channel_idx:channel_idx + 1].to(base_dtype)
                    warped_channel = F.grid_sample(
                        channel_slice,
                        grid,
                        mode="bilinear",
                        padding_mode="reflection",
                        align_corners=True,
                    ).to(sample.dtype)

                    if mix >= 1.0:
                        sample_result[:, channel_idx:channel_idx + 1] = warped_channel
                    else:
                        delta = warped_channel - sample[:, channel_idx:channel_idx + 1]
                        sample_result[:, channel_idx:channel_idx + 1] = sample[:, channel_idx:channel_idx + 1] + delta * mix

            result[idx] = sample_result[0]

        if is_video:
            result = result.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)

        out = latent.copy()
        out["samples"] = result.cpu()

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
                "model": ("MODEL", {"tooltip": "Diffusion model that defines the forward noise schedule."}),
                "latent": ("LATENT", {"tooltip": "Clean latent to push forward along the schedule."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for the forward diffusion noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Number of steps in the sampler's schedule."}),
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
                "conditioning": ("CONDITIONING", {"tooltip": "Conditioning list to perturb with Gaussian noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed that drives the conditioning noise."}),
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


class ConditioningGaussianBlur:
    """Applies Gaussian smoothing along the token dimension of conditioning embeddings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "Conditioning list whose token dimension will be blurred."}),
                "sigma": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Standard deviation of the blur kernel along the token axis."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "blur"
    CATEGORY = "conditioning/filter"

    def blur(self, conditioning, sigma):
        if sigma <= 0.0:
            return (conditioning,)

        result = []

        for embedding, metadata in conditioning:
            if not isinstance(embedding, torch.Tensor) or embedding.dim() < 2:
                result.append([embedding, metadata])
                continue

            needs_batch_dim = embedding.dim() == 2
            embedding_tensor = embedding.unsqueeze(0) if needs_batch_dim else embedding

            conv_dtype = self._select_conv_dtype(embedding_tensor)
            blur_kernel, padding = self._build_kernel(embedding_tensor.device, conv_dtype, sigma)

            original_shape = embedding_tensor.shape
            tokens = original_shape[-2]
            features = original_shape[-1]

            batch = math.prod(original_shape[:-2]) if len(original_shape) > 2 else original_shape[0]
            reshaped = embedding_tensor.reshape(batch, tokens, features).permute(0, 2, 1)

            if reshaped.dtype != conv_dtype:
                reshaped = reshaped.to(conv_dtype)

            kernel = blur_kernel.repeat(features, 1, 1)
            blurred = F.conv1d(reshaped, kernel, padding=padding, groups=features)
            blurred = blurred.permute(0, 2, 1).reshape(original_shape)

            if blurred.dtype != embedding_tensor.dtype:
                blurred = blurred.to(embedding_tensor.dtype)

            if needs_batch_dim:
                blurred = blurred.squeeze(0)

            result.append([blurred, metadata])

        return (result,)

    @staticmethod
    def _build_kernel(device, dtype, sigma):
        kernel_size = int(sigma * 6) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        half = kernel_size // 2
        positions = torch.arange(kernel_size, device=device, dtype=dtype) - half
        kernel = torch.exp(-0.5 * (positions / max(sigma, 1e-6)) ** 2)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, kernel_size)
        return kernel, half

    @staticmethod
    def _select_conv_dtype(tensor):
        if tensor.device.type == "cpu" and tensor.dtype not in (torch.float32, torch.float64):
            return torch.float32
        return tensor.dtype


class ConditioningFrequencySplit:
    """Separates conditioning embeddings into low/high bands via Gaussian smoothing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "Conditioning list to separate into low/high bands."}),
                "sigma": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Cutoff for the Gaussian low-pass applied along the token axis."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("low_pass", "high_pass")
    FUNCTION = "split"
    CATEGORY = "conditioning/filter"

    def split(self, conditioning, sigma):
        if sigma <= 0.0:
            low_list = [self._clone_entry(item) for item in conditioning]
            high_list = [self._zero_entry(item) for item in conditioning]
            return (low_list, high_list)

        low_list = []
        high_list = []

        for embedding, metadata in conditioning:
            if not isinstance(embedding, torch.Tensor) or embedding.dim() < 2:
                low_list.append([embedding, dict(metadata)])
                high_list.append(self._zero_entry([embedding, metadata]))
                continue

            needs_batch_dim = embedding.dim() == 2
            embedding_tensor = embedding.unsqueeze(0) if needs_batch_dim else embedding

            conv_dtype = ConditioningGaussianBlur._select_conv_dtype(embedding_tensor)
            kernel, padding = ConditioningGaussianBlur._build_kernel(embedding_tensor.device, conv_dtype, sigma)

            original_shape = embedding_tensor.shape
            tokens = original_shape[-2]
            features = original_shape[-1]

            batch = math.prod(original_shape[:-2]) if len(original_shape) > 2 else original_shape[0]
            reshaped = embedding_tensor.reshape(batch, tokens, features).permute(0, 2, 1)

            if reshaped.dtype != conv_dtype:
                reshaped = reshaped.to(conv_dtype)

            kernel = kernel.repeat(features, 1, 1)
            blurred = F.conv1d(reshaped, kernel, padding=padding, groups=features)
            blurred = blurred.permute(0, 2, 1).reshape(original_shape)

            if blurred.dtype != embedding_tensor.dtype:
                blurred = blurred.to(embedding_tensor.dtype)

            if needs_batch_dim:
                blurred = blurred.squeeze(0)

            high_embedding = embedding - blurred

            low_meta = dict(metadata)
            high_meta = dict(metadata)

            pooled_output = low_meta.get("pooled_output")
            if isinstance(pooled_output, torch.Tensor):
                high_meta["pooled_output"] = torch.zeros_like(pooled_output)

            low_list.append([blurred, low_meta])
            high_list.append([high_embedding, high_meta])

        return (low_list, high_list)

    @staticmethod
    def _clone_entry(entry):
        embedding, metadata = entry
        return [embedding, dict(metadata)]

    @staticmethod
    def _zero_entry(entry):
        embedding, metadata = entry
        if isinstance(embedding, torch.Tensor):
            zero_embedding = torch.zeros_like(embedding)
        else:
            zero_embedding = embedding

        meta_copy = dict(metadata)
        pooled_output = meta_copy.get("pooled_output")
        if isinstance(pooled_output, torch.Tensor):
            meta_copy["pooled_output"] = torch.zeros_like(pooled_output)
        return [zero_embedding, meta_copy]


class ConditioningFrequencyMerge:
    """Recombines low/high conditioning bands back into a single conditioning list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low_pass": ("CONDITIONING", {"tooltip": "Low-frequency conditioning list produced by the split node."}),
                "high_pass": ("CONDITIONING", {"tooltip": "High-frequency conditioning list to recombine."}),
                "low_gain": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Multiplier for the low-pass band before merging."
                }),
                "high_gain": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Multiplier for the high-pass band before merging."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "merge"
    CATEGORY = "conditioning/filter"

    def merge(self, low_pass, high_pass, low_gain, high_gain):
        if len(low_pass) != len(high_pass):
            raise ValueError("Low-pass and high-pass conditioning lists must have the same length")

        merged = []

        for low_item, high_item in zip(low_pass, high_pass):
            low_embedding, low_meta = low_item
            high_embedding, high_meta = high_item

            if not isinstance(low_embedding, torch.Tensor) or not isinstance(high_embedding, torch.Tensor):
                merged.append([low_embedding, dict(low_meta)])
                continue

            combined_embedding = low_embedding * low_gain + high_embedding * high_gain

            new_metadata = dict(low_meta)
            low_pooled = low_meta.get("pooled_output")
            high_pooled = high_meta.get("pooled_output") if isinstance(high_meta, dict) else None

            if isinstance(low_pooled, torch.Tensor) or isinstance(high_pooled, torch.Tensor):
                low_pooled_tensor = low_pooled if isinstance(low_pooled, torch.Tensor) else torch.zeros_like(high_pooled)
                high_pooled_tensor = high_pooled if isinstance(high_pooled, torch.Tensor) else torch.zeros_like(low_pooled_tensor)
                new_metadata["pooled_output"] = low_pooled_tensor * low_gain + high_pooled_tensor * high_gain

            merged.append([combined_embedding, new_metadata])

        return (merged,)


class ConditioningScale:
    """Scales conditioning embeddings (and pooled outputs) to amplify or mute prompt influence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "Conditioning list to scale."}),
                "factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Multiplier applied to embeddings. 0.0 mutes, 1.0 keeps original strength."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "scale"
    CATEGORY = "conditioning/filter"

    def scale(self, conditioning, factor):
        if factor == 1.0:
            return (conditioning,)

        result = []

        for embedding, metadata in conditioning:
            if not isinstance(embedding, torch.Tensor):
                result.append([embedding, metadata])
                continue

            scaled_embedding = embedding * factor
            new_metadata = dict(metadata)

            pooled_output = new_metadata.get("pooled_output")
            if isinstance(pooled_output, torch.Tensor):
                new_metadata["pooled_output"] = pooled_output * factor

            result.append([scaled_embedding, new_metadata])

        return (result,)

NODE_CLASS_MAPPINGS = {
    "QwenRectifiedFlowInverter": QwenRectifiedFlowInverter,
    "LatentGaussianBlur": LatentGaussianBlur,
    "LatentFrequencySplit": LatentFrequencySplit,
    "LatentAddNoise": LatentAddNoise,
    "LatentPerlinFractalNoise": LatentPerlinFractalNoise,
    "LatentSimplexNoise": LatentSimplexNoise,
    "LatentWorleyNoise": LatentWorleyNoise,
    "LatentReactionDiffusion": LatentReactionDiffusion,
    "LatentFractalBrownianMotion": LatentFractalBrownianMotion,
    "LatentSwirlNoise": LatentSwirlNoise,
    "LatentForwardDiffusion": LatentForwardDiffusion,
    "LatentHybridInverter": LatentHybridInverter,
    "ConditioningAddNoise": ConditioningAddNoise,
    "ConditioningGaussianBlur": ConditioningGaussianBlur,
    "ConditioningFrequencySplit": ConditioningFrequencySplit,
    "ConditioningFrequencyMerge": ConditioningFrequencyMerge,
    "ConditioningScale": ConditioningScale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenRectifiedFlowInverter": "Qwen Rectified Flow Inverter",
    "LatentGaussianBlur": "Latent Gaussian Blur",
    "LatentFrequencySplit": "Latent Frequency Split",
    "LatentAddNoise": "Add Latent Noise (Seeded)",
    "LatentPerlinFractalNoise": "Latent Perlin Fractal Noise",
    "LatentSimplexNoise": "Latent Simplex Noise",
    "LatentWorleyNoise": "Latent Worley Noise",
    "LatentReactionDiffusion": "Latent Reaction-Diffusion",
    "LatentFractalBrownianMotion": "Latent Fractal Brownian Motion",
    "LatentSwirlNoise": "Latent Swirl Noise",
    "LatentForwardDiffusion": "Forward Diffusion (Add Scheduled Noise)",
    "LatentHybridInverter": "Latent Hybrid Inverter (Qwen)",
    "ConditioningAddNoise": "Conditioning (Add Noise)",
    "ConditioningGaussianBlur": "Conditioning (Gaussian Blur)",
    "ConditioningFrequencySplit": "Conditioning (Frequency Split)",
    "ConditioningFrequencyMerge": "Conditioning (Frequency Merge)",
    "ConditioningScale": "Conditioning (Scale)",
}
