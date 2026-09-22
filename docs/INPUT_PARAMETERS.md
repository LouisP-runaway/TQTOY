# Input parameters

Every parameter of the `tqtoy run` input file. The same list is printed by `tqtoy params` (add
a word to filter it, e.g. `tqtoy params temperature`), and the table below by
`tqtoy params --markdown`. A commented file holding every parameter is written by `tqtoy init`.

Parameters are written in YAML with their section, e.g. `plasma: {Te0: 10.0e3}`, or on the
command line with their dotted path, e.g. `--set plasma.Te0=10e3`. Any scalar parameter can be
a scan axis (see the last row of the table, and section 6 of [TUTORIAL.md](TUTORIAL.md)).

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `plasma.Te0` | float | `20000` | eV | Initial electron temperature |
| `plasma.Ti0` | float or null | `null` | eV | Initial ion temperature (null: equal to Te0) |
| `plasma.ne0` | float | `1e+20` | m^-3 | Initial electron density |
| `plasma.ni0` | float or null | `null` | m^-3 | Initial main-ion density (null: equal to ne0) |
| `plasma.J` | float | `1e+06` | A m^-2 | Current density |
| `plasma.J_final` | float or null | `null` | A m^-2 | Optional linear current decay: J reaches J_final at t = J_ramp_time |
| `plasma.J_ramp_time` | float or null | `null` | s | Duration of the linear current decay (with J_final) |
| `plasma.B` | float | `5.3` | T | Magnetic field (used by the ablation law only) |
| `plasma.cold_T0` | float | `2` | eV | Initial electron and ion temperature of the cold population (2 populations) |
| `plasma.cold_n0` | float | `1e+10` | m^-3 | Initial electron and ion density of the cold population (2 populations) |
| `impurities.<element>.injected_atom_density` | float | `0` | m^-3 | Density of atoms of this element deposited by the injection, summed over all charge states |
| `impurities.<element>.background_density` | float | `0` | m^-3 | Density already present at t = 0 |
| `injection.source` | str | `prescribed` |  | prescribed (constant deposition rate) or ablation (pellet ablation law) |
| `injection.t_start` | float | `1e-07` | s | Start of the injection |
| `injection.duration` | float | `1e-05` | s | Duration of the injection |
| `injection.n_D` | float | `1e+22` | m^-3 | Deposited deuterium atom density (prescribed source) |
| `injection.charge_state` | int | `0` |  | Charge state of the injected and background impurities (0: neutral) |
| `injection.ablation.D2_fraction` | float or null | `null` |  | Molecular fraction X of D2 in the pellet (null: X = (n_D/2) / (n_D/2 + injected_atom_density)) |
| `injection.ablation.volume` | float | `840` | m^3 | Volume where the ablated material is deposited |
| `injection.ablation.shard_radius` | float | `0.002` | m | Radius of identical shards (ignored if parks is set) |
| `injection.ablation.n_shards` | int | `300` |  | Number of shards |
| `injection.ablation.deuterium_coefficients` | bool | `false` |  | Use the D coefficients of the ablation law (false: the H coefficients) |
| `injection.ablation.parks` | section or null | `null` |  | Sample the shard radii from the Parks distribution (null: identical shards) |
| `injection.ablation.parks.n_atoms` | float | `1.8e+24` |  | Total number of atoms in the pellet |
| `injection.ablation.parks.solid_density` | float | `4.95e+28` | m^-3 | Atomic density of the solid pellet |
| `injection.ablation.parks.r_min` | float | `0.0001` | m | Lower bound of the sampled shard radius |
| `injection.ablation.parks.r_max` | float | `0.01` | m | Upper bound of the sampled shard radius |
| `injection.ablation.seed` | int or null | `null` |  | Random seed of the Parks sampling (integer: reproducible radii) |
| `injection.ablation.switch_on_time` | float | `1e-06` | s | Time constant of the tanh switch-on of the ablation |
| `model.populations` | int | `1` |  | 1: single thermal population, 2: hot and cold populations |
| `model.block_electron_ion_exchange` | bool | `false` |  | Switch off the electron-ion energy exchange |
| `model.resistivity` | str | `spitzer` |  | spitzer or multi_species (2 populations only, not validated) |
| `model.current_sharing` | str | `density_weighted` |  | How the current is shared between the two populations (spitzer resistivity only). density_weighted: the conductivity of a population is weighted by its own electron density, sigma_p = (n_e,p / n_e,total) / eta(T_p). spitzer: two parallel resistors eta(T_hot) and eta(T_cold), which gives a population emptied of its electrons a finite share of the current (see docs/MODEL_EQUATIONS.pdf) |
| `model.atomic_rate_multiplier` | float | `1` |  | Factor on ionisation and recombination rates (>> 1: coronal limit) |
| `model.linear_loss_coefficient` | float | `0` | W m^-3 eV^-1 | Extra loss P = coef * T, off by default |
| `model.main_ion_mass` | float | `1` |  | Main-ion mass in the collision times, in proton masses (1 also for a deuterium plasma, see docs/MODEL_EQUATIONS.pdf) |
| `model.stochastic.enabled` | bool | `true` |  | Include the stochastic loss |
| `model.stochastic.deltaB_over_B` | float | `0.01` |  | Relative magnetic perturbation |
| `model.stochastic.r` | float | `0.6` | m | Radius where the loss is evaluated (r/a < 1/sqrt(6)) |
| `model.stochastic.a` | float | `2` | m | Minor radius |
| `model.stochastic.coulomb_log` | float | `15` |  | Coulomb logarithm of the stochastic loss (constant) |
| `model.radiation_limit.enabled` | bool | `true` |  | Limit the radiated power below Te_below |
| `model.radiation_limit.Te_below` | float | `100` | eV | Temperature below which the limiter applies |
| `model.radiation_limit.time_scale` | float | `1e-06` | s | The radiated power cannot exceed 1.5 ne Te e / time_scale |
| `time.t_end` | float | `0.01` | s | Last output time |
| `time.n_output` | int | `500` |  | Number of output times |
| `time.spacing` | str | `log` |  | log or linear |
| `time.t_start` | float or null | `null` | s | First output time (null: injection.t_start) |
| `solver.method` | str | `Radau` |  | Integration method (Radau, BDF, LSODA, RK45, RK23, DOP853) |
| `solver.rtol` | float | `0.01` |  | Relative tolerance (check the convergence by reducing it) |
| `solver.atol` | float | `1e-10` |  | Absolute tolerance on the normalised state vector |
| `solver.state_variables` | str | `temperature` |  | What the solver integrates for the four temperatures. temperature: T itself, whose equation dT/dt = (2/3e) P/n - T (dn/dt)/n carries a 1/n. energy: the energy densities n_e T_e and n_i T_i, whose equation d(nT)/dt = (2/3e) P carries none. The two are the same equations, one multiplied by n, and outputs are temperatures in both cases. The energy form is the regular one but the harder one to integrate, because n T spans 14 decades against 2 for T, so it needs atol around 1e-13 |
| `solver.floor` | float | `0.01` |  | Floor on temperatures [eV] and densities [m^-3] |
| `solver.jacobian` | str | `numeric` |  | How the implicit methods get the Jacobian. numeric: by differences, as scipy does by default. analytic: the impurity charge-state columns are differentiated analytically and only the eight fluid columns are differenced, which removes most of the right-hand-side evaluations of the Jacobian. The Jacobian drives the Newton iteration and the step size, not the solution of the collocation equations |
| `output.file` | str | `results/tqtoy_results.h5` |  | HDF5 output file, created if its directory does not exist (tqtoy run -o overrides it) |
| `output.energy_integrals` | bool | `true` |  | Integrate the energy terms on the internal solver steps |
| `output.energy_refinement` | int | `8` |  | Sub-intervals per solver step for these integrals |
| `fhte.enabled` | bool | `false` |  | Run the FHTE on every point at the end of tqtoy run (also: tqtoy fhte results.h5) |
| `fhte.n_output` | int | `2` |  | Number of evaluation times from the first to the last output time (the last one gives the final runaway fraction). With output_window = active and only two times, the second one is the end of the runaway window rather than the last time of the run |
| `fhte.output_window` | str | `active` |  | Where the evaluation times are placed. active: inside the time windows where E > E_c, the only ones where runaways are created (no backward integration outside, the runaway population is kept constant there). full: over the whole run, with a backward integration at every time |
| `fhte.output_spacing` | str | `linear` |  | Spacing of the evaluation times: linear or log |
| `fhte.accumulate` | bool | `true` |  | Keep the runaway population once it is created: the reported fraction is the running maximum over the evaluation times. The estimator gives, at each time, the fraction of the initial distribution that is above the critical momentum at that time; this fraction falls when E/E_c decreases, although the electrons that already ran away are not lost. false: report the instantaneous fraction at each evaluation time |
| `fhte.n_trajectory` | int | `2` |  | Stored points per backward momentum trajectory |
| `fhte.trajectory_spacing` | str | `linear` |  | Spacing of the stored trajectory points: linear or log |
| `fhte.field` | str | `spitzer` |  | Parallel electric field: spitzer (from J with zeff and neoclassical_factor) or model (field of the TQ model, TQ runs only) |
| `fhte.zeff` | float | `1` |  | Effective charge in the Spitzer field of the FHTE |
| `fhte.neoclassical_factor` | float | `2` |  | Factor on the Spitzer resistivity of the FHTE (neoclassical correction) |
| `fhte.chandrasekhar` | bool | `false` |  | Multiply the collisional drag by the Chandrasekhar function |
| `fhte.temperature` | str | `average` |  | Bulk temperature from a TQ run: average (density weighted), cold or hot |
| `fhte.bound_electron_weight` | float | `0` |  | Weight of the electrons bound to impurities in the collisional density (0: free electrons only) |
| `fhte.T_floor` | float | `10` | eV | Floor on the bulk temperature |
| `fhte.n_floor` | float | `1e+15` | m^-3 | Floor on the density |
| `fhte.critical_iterations` | int | `4` |  | Fixed-point iterations on gamma_c for the critical momentum |
| `fhte.rtol` | float | `1e-05` |  | Relative tolerance of the backward momentum integration |
| `fhte.atol` | float | `1e-10` |  | Absolute tolerance of the backward momentum integration |
| `fhte.gamma_max` | float | `4` |  | Upper bound of the Maxwell-Juttner integration in gamma |
| `fhte.gamma_points` | int | `200000` |  | Number of points of the Maxwell-Juttner integration |
| `scan` | mapping | `{}` |  | Scan axes: {dotted.path: values}. Values: list, {linspace: [a, b, n]}, {geomspace: [a, b, n]} or {logspace: [exp_a, exp_b, n]} |
