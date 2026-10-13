# 1D convection-diffusion (Python)

A small finite-volume exercise for the periodic 1D advection-diffusion equation. Mostly to test how the spatial scheme affects the answer.

- Upwind or central face fluxes for convection
- Central difference for diffusion, forward Euler in time
- Travelling sine-wave exact solution and a grid error check
- Basic `unittest` tests (mass, constant field, input checks, refinement)

Needs Python 3 and NumPy.

```bash
python -m pip install numpy
python -m unittest -v
python solver.py --study --scheme upwind
python solver.py --study --scheme central
```

The explicit timestep is conservative, but this is still a simple learning solver. For central convection we require nonzero diffusion; pure central advection with forward Euler would be unstable. Don't use it as a general purpose CFD package.
