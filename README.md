# ComfyUI Qwen Rectified Flow Inverter

Custom nodes for ComfyUI that target rectified-flow models (for example, Qwen). This pack focuses on inversion and related diagnostics.

## Nodes

### Qwen Rectified Flow Inverter

Runs the rectified-flow ODE backwards to re-noise an encoded latent. Supports deterministic amplification, seeded stochastic perturbation, and optional output normalisation.

### Latent Hybrid Inverter (Qwen)

Convenience node that runs the inverter plus an internal forward-diffusion anchor pass, then blends their outputs via spherical interpolation.

### Memory Diagnostics (Pass-Through)

Passes data through unchanged while logging basic GPU memory stats.

## Noise nodes moved

All latent/image/conditioning noise + filtering nodes that used to ship here have been moved to `Skoogeer-Noise` (see `../Skoogeer-Noise`).

## Installation

1. Clone or download this repository into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI to load the nodes (look under `Qwen/Sampling` and `Qwen/Diagnostics`).

## Running the tests

The repository ships with a helper script that prepares a virtual environment and executes the pytest suite. From the project root run:

```bash
./run_tests.sh
```

The script creates (or reuses) a `.venv` folder, installs the dependencies from `requirements.txt`, and then launches `pytest`.
