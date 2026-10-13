"""Small 1D advection-diffusion solver with periodic boundaries."""

import argparse
import numpy as np


def step(phi, dx, dt, velocity, diffusion, scheme="upwind"):
    """One finite-volume forward Euler step (upwind flux for advection)."""
    if dx <= 0 or dt <= 0 or diffusion < 0:
        raise ValueError('dx and dt must be positive; diffusion cannot be negative')
    if scheme not in ("upwind", "central"):
        raise ValueError("scheme must be upwind or central")
    if scheme == "central" and velocity != 0 and diffusion == 0:
        raise ValueError("central advection with forward Euler is unstable without diffusion")
    if len(phi) < 3:
        raise ValueError('need at least 3 cells')

    left = np.roll(phi, 1)
    right = np.roll(phi, -1)
    # Use upstream value at each face, depending on flow direction.
    if scheme == "central":
        flux_right = velocity * (phi + right) / 2
        flux_left = velocity * (left + phi) / 2
    elif velocity >= 0:
        flux_right = velocity * phi
        flux_left = velocity * left
    else:
        flux_right = velocity * right
        flux_left = velocity * phi
    convection = -(flux_right - flux_left) / dx
    diffusion_term = diffusion * (right - 2 * phi + left) / dx**2
    return phi + dt * (convection + diffusion_term)


def solve(cells=100, velocity=1.0, diffusion=0.01, final_time=0.1, scheme="upwind"):
    if cells < 3 or diffusion < 0 or final_time < 0:
        raise ValueError('invalid grid, diffusion, or final time')
    if scheme not in ("upwind", "central"):
        raise ValueError("unknown scheme")
    if scheme == "central" and velocity != 0 and diffusion == 0:
        raise ValueError("central Euler advection needs diffusion")
    dx = 1.0 / cells
    x = (np.arange(cells) + 0.5) * dx
    phi = np.sin(2 * np.pi * x)
    if final_time == 0:
        return x, phi
    adv_dt = dx / abs(velocity) if velocity else float('inf')
    diff_dt = dx**2 / (2 * diffusion) if diffusion else float('inf')
    dt_limit = min(adv_dt, diff_dt)
    if scheme == "central" and velocity != 0:
        # Extra FTCS stability restriction when convection uses a central flux.
        dt_limit = min(dt_limit, 2 * diffusion / velocity**2)
    if not np.isfinite(dt_limit):
        return x, phi
    count = max(1, int(np.ceil(final_time / (0.4 * dt_limit))))
    dt = final_time / count
    for _ in range(count):
        phi = step(phi, dx, dt, velocity, diffusion, scheme)
    return x, phi


def exact_solution(x, time, velocity=1.0, diffusion=0.01):
    """Exact travelling sine wave on the periodic unit domain."""
    return np.exp(-4 * np.pi**2 * diffusion * time) * np.sin(
        2 * np.pi * (x - velocity * time)
    )


def grid_study(scheme="upwind", sizes=(25, 50, 100, 200), time=0.1):
    """Print an L2 error table so we can see whether refinement helps."""
    errors = []
    for n in sizes:
        x, phi = solve(cells=n, final_time=time, scheme=scheme)
        err = np.sqrt(np.mean((phi - exact_solution(x, time))**2))
        errors.append(err)
        print(f"{n:5d} cells  L2 error = {err:.6e}")
    for i in range(1, len(errors)):
        order = np.log(errors[i-1] / errors[i]) / np.log(sizes[i] / sizes[i-1])
        print(f"{sizes[i-1]} -> {sizes[i]}: observed order {order:.2f}")
    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, default=100)
    parser.add_argument('--time', type=float, default=0.1)
    parser.add_argument('--scheme', choices=['upwind', 'central'], default='upwind')
    parser.add_argument('--study', action='store_true')
    args = parser.parse_args()
    if args.study:
        grid_study(scheme=args.scheme, time=args.time)
    else:
        x, result = solve(cells=args.cells, final_time=args.time, scheme=args.scheme)
        error = np.sqrt(np.mean((result - exact_solution(x, args.time))**2))
        print(f'cells={len(x)}, min={result.min():.6f}, max={result.max():.6f}, L2={error:.4e}')
