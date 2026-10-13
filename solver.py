"""Small 1D advection-diffusion solver with periodic boundaries."""

import argparse
import numpy as np


def step(phi, dx, dt, velocity, diffusion):
    """One finite-volume forward Euler step (upwind flux for advection)."""
    if dx <= 0 or dt <= 0 or diffusion < 0:
        raise ValueError('dx and dt must be positive; diffusion cannot be negative')
    if len(phi) < 3:
        raise ValueError('need at least 3 cells')

    left = np.roll(phi, 1)
    right = np.roll(phi, -1)
    # Use upstream value at each face, depending on flow direction.
    if velocity >= 0:
        flux_right = velocity * phi
        flux_left = velocity * left
    else:
        flux_right = velocity * right
        flux_left = velocity * phi
    convection = -(flux_right - flux_left) / dx
    diffusion_term = diffusion * (right - 2 * phi + left) / dx**2
    return phi + dt * (convection + diffusion_term)


def solve(cells=100, velocity=1.0, diffusion=0.01, final_time=0.1):
    if cells < 3 or diffusion < 0 or final_time < 0:
        raise ValueError('invalid grid, diffusion, or final time')
    dx = 1.0 / cells
    x = (np.arange(cells) + 0.5) * dx
    phi = np.sin(2 * np.pi * x)
    if final_time == 0:
        return x, phi
    adv_dt = dx / abs(velocity) if velocity else float('inf')
    diff_dt = dx**2 / (2 * diffusion) if diffusion else float('inf')
    dt_limit = min(adv_dt, diff_dt)
    if not np.isfinite(dt_limit):
        return x, phi
    count = max(1, int(np.ceil(final_time / (0.4 * dt_limit))))
    dt = final_time / count
    for _ in range(count):
        phi = step(phi, dx, dt, velocity, diffusion)
    return x, phi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, default=100)
    parser.add_argument('--time', type=float, default=0.1)
    args = parser.parse_args()
    x, result = solve(cells=args.cells, final_time=args.time)
    print(f'cells={len(x)}, min={result.min():.6f}, max={result.max():.6f}')
