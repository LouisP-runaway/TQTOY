"""Use tqtoy from Python: one run, a small scan and a few figures.

Run from the repository root:  python examples/api_example.py
"""

import matplotlib
import numpy as np

from tqtoy import Config, simulate
from tqtoy.diagnostics import compute_terms, summary
from tqtoy.io import load_results, save_results
from tqtoy.plotting import make_figures, plot_scan_map
from tqtoy.scan import run_scan

matplotlib.use("Agg")


def main():
    # One run: start from the defaults and change a few parameters
    cfg = Config().with_overrides({"plasma.Te0": 10e3, "impurities.neon.injected_atom_density": 2e19,
                                   "time.n_output": 300})
    run = simulate(cfg)
    print(f"status: {run.status}, final <Te> = {run.Te_avg[-1]:.2f} eV")
    for key, value in summary(run).items():
        print(f"  {key:28s} {value:.4g}")

    # All intermediate terms of the model along the run (powers, currents, Zeff, ...)
    terms = compute_terms(run)
    print("peak radiated power density [MW/m3]:", np.max(terms["P_rad_cold"]) * 1e-6)
    make_figures(run, ["temperatures", "powers", "energy"], outdir="figures_api")

    # A small scan, run on 2 processes and saved to HDF5
    scan_cfg = cfg.with_overrides({"scan.injection.n_D": {"logspace": [20, 23, 4]},
                                   "scan.impurities.neon.injected_atom_density": [1e18, 1e19, 1e20]})
    results = run_scan(scan_cfg, jobs=2)
    save_results(results, "figures_api/scan.h5")

    results = load_results("figures_api/scan.h5")
    one = results.run(n_D=1e22, injected_atom_density=1e19)      # nearest scan point
    print("selected point:", one.config.injection.n_D, one.config.impurities["neon"].injected_atom_density)
    plot_scan_map(results, "t_TQ").savefig("figures_api/map_t_TQ.png", dpi=150)


if __name__ == "__main__":      # required for parallel scans on Windows and macOS
    main()
