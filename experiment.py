"""Run a few checks and save figures/CSV for later comparison."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver import solve, exact_solution, sharp_profile, time_step_study


def main():
    out = Path('results')
    out.mkdir(exist_ok=True)
    rows = []
    fig, ax = plt.subplots()
    for scheme in ('upwind', 'central'):
        for n in (50, 100, 200, 400):
            x, phi = solve(cells=n, scheme=scheme, cfl=0.1)
            error = np.sqrt(np.mean((phi - exact_solution(x, 0.1))**2))
            rows.append((scheme, n, float(error)))
        x, phi = solve(cells=200, scheme=scheme, cfl=0.1)
        ax.plot(x, phi, label=scheme)
    ax.plot(x, exact_solution(x, 0.1), '--', label='exact')
    ax.set(xlabel='x', ylabel='scalar', title='Smooth periodic solution')
    ax.legend()
    fig.savefig(out / 'smooth.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    for scheme in ('upwind', 'central'):
        x, phi = solve(cells=200, scheme=scheme, initial=sharp_profile,
                       final_time=0.08, diffusion=0.01, cfl=0.1)
        ax.plot(x, phi, label=scheme)
    ax.plot(x, sharp_profile((x-0.08) % 1), '--', label='advected initial shape')
    ax.set(xlabel='x', ylabel='scalar', title='Sharp profile (reference ignores diffusion)')
    ax.legend()
    fig.savefig(out / 'sharp.png', dpi=160)
    plt.close(fig)

    with (out / 'grid_errors.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('scheme', 'cells', 'L2_error'))
        writer.writerows(rows)
    with (out / 'timestep_errors.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('scheme', 'cfl_factor', 'L2_error'))
        for scheme in ('upwind', 'central'):
            writer.writerows((scheme, cf, err) for cf, err in time_step_study(scheme=scheme))
    for row in rows:
        print(f'{row[0]:7s} {row[1]:4d} cells  L2={row[2]:.6g}')
    print('Wrote results/*.png and results/*.csv')


if __name__ == '__main__':
    main()
