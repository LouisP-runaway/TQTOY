"""Simulation configuration: typed dataclasses, YAML loading, validation and scans.

A configuration is a nested set of dataclasses. It can be written as a YAML file
(see ``tqtoy init``) or built in Python. Every scalar parameter can be scanned
with the ``scan`` section, using its dotted path, e.g. ``injection.n_D``.
"""

from __future__ import annotations

import dataclasses
import itertools
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


class ConfigError(ValueError):
    """Invalid configuration."""


def parse_yaml(text: str, name: str = "input file"):
    """Read YAML text, tolerating tabulations and reporting syntax errors with their position.

    YAML forbids tabulations in the indentation. Editors insert them anyway, so the leading
    tabulations of each line are expanded to spaces on the standard eight-column tab stops
    (what the file looks like in an editor) instead of making the file unreadable. Tabulations
    inside a value are legal YAML and are left untouched.
    """
    if "\t" in text:
        lines = []
        for line in text.splitlines():
            body = line.lstrip("\t ")
            lines.append(line[:len(line) - len(body)].expandtabs() + body)
        text = "\n".join(lines)
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise ConfigError(f"{name}: invalid YAML{where}: {problem}. Check the indentation "
                          f"(spaces, two per level) and the ':' after each entry name.") from None


def read_yaml(path):
    """Read a YAML file (see :func:`parse_yaml`)."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_yaml(f.read(), str(path))


SOLVER_METHODS = ("Radau", "BDF", "LSODA", "RK45", "RK23", "DOP853")
IMPLICIT_METHODS = ("Radau", "BDF", "LSODA")      # the only ones that use a Jacobian


# --------------------------------------------------------------------------- sections
# Each parameter carries its unit and a short description in the field metadata.
# They are printed by ``tqtoy params`` and written to docs/INPUT_PARAMETERS.md.

def _p(default, help: str, unit: str = ""):
    """Scalar parameter with documentation."""
    return field(default=default, metadata={"help": help, "unit": unit})


def _section(factory, help: str):
    """Sub-section with documentation."""
    return field(default_factory=factory, metadata={"help": help, "unit": ""})


@dataclass
class PlasmaConfig:
    """Pre-disruption plasma (initial state of the hot population)."""

    Te0: float = _p(20.0e3, "Initial electron temperature", "eV")
    Ti0: Optional[float] = _p(None, "Initial ion temperature (null: equal to Te0)", "eV")
    ne0: float = _p(1.0e20, "Initial electron density", "m^-3")
    ni0: Optional[float] = _p(None, "Initial main-ion density (null: equal to ne0)", "m^-3")
    J: float = _p(1.0e6, "Current density", "A m^-2")
    J_final: Optional[float] = _p(None, "Optional linear current decay: J reaches J_final at t = J_ramp_time",
                                  "A m^-2")
    J_ramp_time: Optional[float] = _p(None, "Duration of the linear current decay (with J_final)", "s")
    B: float = _p(5.3, "Magnetic field (used by the ablation law only)", "T")
    cold_T0: float = _p(2.0, "Initial electron and ion temperature of the cold population (2 populations)",
                        "eV")
    cold_n0: float = _p(1.0e10, "Initial electron and ion density of the cold population (2 populations)",
                        "m^-3")


@dataclass
class ImpurityConfig:
    """One impurity element (the key is a CHERAB element name or symbol: neon, Ne, argon...)."""

    injected_atom_density: float = _p(0.0, "Density of atoms of this element deposited by the "
                                           "injection, summed over all charge states", "m^-3")
    background_density: float = _p(0.0, "Density already present at t = 0", "m^-3")


@dataclass
class ParksConfig:
    """Parks shard-size distribution (ablation source only)."""

    n_atoms: float = _p(1.8e24, "Total number of atoms in the pellet")
    solid_density: float = _p(4.95e28, "Atomic density of the solid pellet", "m^-3")
    r_min: float = _p(1.0e-4, "Lower bound of the sampled shard radius", "m")
    r_max: float = _p(1.0e-2, "Upper bound of the sampled shard radius", "m")


@dataclass
class AblationConfig:
    """Shattered-pellet ablation source (injection.source = ablation)."""

    D2_fraction: Optional[float] = _p(None, "Molecular fraction X of D2 in the pellet "
                                            "(null: X = (n_D/2) / (n_D/2 + injected_atom_density))")
    volume: float = _p(840.0, "Volume where the ablated material is deposited", "m^3")
    shard_radius: float = _p(2.0e-3, "Radius of identical shards (ignored if parks is set)", "m")
    n_shards: int = _p(300, "Number of shards")
    deuterium_coefficients: bool = _p(False, "Use the D coefficients of the ablation law "
                                             "(false: the H coefficients)")
    parks: Optional[ParksConfig] = _p(None, "Sample the shard radii from the Parks distribution "
                                            "(null: identical shards)")
    seed: Optional[int] = _p(None, "Random seed of the Parks sampling (integer: reproducible radii)")
    switch_on_time: float = _p(1.0e-6, "Time constant of the tanh switch-on of the ablation", "s")


@dataclass
class InjectionConfig:
    """Material injection."""

    source: str = _p("prescribed", "prescribed (constant deposition rate) or ablation (pellet ablation law)")
    t_start: float = _p(1.0e-7, "Start of the injection", "s")
    duration: float = _p(1.0e-5, "Duration of the injection", "s")
    n_D: float = _p(1.0e22, "Deposited deuterium atom density (prescribed source)", "m^-3")
    charge_state: int = _p(0, "Charge state of the injected and background impurities (0: neutral)")
    ablation: AblationConfig = _section(AblationConfig, "Ablation source parameters")


@dataclass
class StochasticConfig:
    """Electron heat loss along stochastic field lines."""

    enabled: bool = _p(True, "Include the stochastic loss")
    deltaB_over_B: float = _p(1.0e-2, "Relative magnetic perturbation")
    r: float = _p(0.6, "Radius where the loss is evaluated (r/a < 1/sqrt(6))", "m")
    a: float = _p(2.0, "Minor radius", "m")
    coulomb_log: float = _p(15.0, "Coulomb logarithm of the stochastic loss (constant)")


@dataclass
class RadiationLimitConfig:
    """Limiter of the radiated power at low temperature."""

    enabled: bool = _p(True, "Limit the radiated power below Te_below")
    Te_below: float = _p(100.0, "Temperature below which the limiter applies", "eV")
    time_scale: float = _p(1.0e-6, "The radiated power cannot exceed 1.5 ne Te e / time_scale", "s")


@dataclass
class ModelConfig:
    """Physics options."""

    populations: int = _p(1, "1: single thermal population, 2: hot and cold populations")
    block_electron_ion_exchange: bool = _p(False, "Switch off the electron-ion energy exchange")
    resistivity: str = _p("spitzer", "spitzer or multi_species (2 populations only, not validated)")
    current_sharing: str = _p("density_weighted", "How the current is shared between the two "
                                                 "populations (spitzer resistivity only). "
                                                 "density_weighted: the conductivity of a population "
                                                 "is weighted by its own electron density, "
                                                 "sigma_p = (n_e,p / n_e,total) / eta(T_p). spitzer: "
                                                 "two parallel resistors eta(T_hot) and eta(T_cold), "
                                                 "which gives a population emptied of its electrons "
                                                 "a finite share of the current "
                                                 "(see docs/MODEL_EQUATIONS.pdf)")
    atomic_rate_multiplier: float = _p(1.0, "Factor on ionisation and recombination rates "
                                            "(>> 1: coronal limit)")
    linear_loss_coefficient: float = _p(0.0, "Extra loss P = coef * T, off by default", "W m^-3 eV^-1")
    main_ion_mass: float = _p(1.0, "Main-ion mass in the collision times, in proton masses "
                                   "(1 also for a deuterium plasma, see docs/MODEL_EQUATIONS.pdf)")
    stochastic: StochasticConfig = _section(StochasticConfig, "Stochastic transport loss")
    radiation_limit: RadiationLimitConfig = _section(RadiationLimitConfig, "Low-temperature radiation limiter")


@dataclass
class TimeConfig:
    """Output time grid."""

    t_end: float = _p(1.0e-2, "Last output time", "s")
    n_output: int = _p(500, "Number of output times")
    spacing: str = _p("log", "log or linear")
    t_start: Optional[float] = _p(None, "First output time (null: injection.t_start)", "s")


@dataclass
class SolverConfig:
    """Options passed to scipy.integrate.solve_ivp."""

    method: str = _p("Radau", "Integration method (Radau, BDF, LSODA, RK45, RK23, DOP853)")
    rtol: float = _p(1.0e-2, "Relative tolerance (check the convergence by reducing it)")
    atol: float = _p(1.0e-10, "Absolute tolerance on the normalised state vector")
    state_variables: str = _p("temperature", "What the solver integrates for the four "
                                             "temperatures. temperature: T itself, whose equation "
                                             "dT/dt = (2/3e) P/n - T (dn/dt)/n carries a 1/n. "
                                             "energy: the energy densities n_e T_e and n_i T_i, "
                                             "whose equation d(nT)/dt = (2/3e) P carries none. The "
                                             "two are the same equations, one multiplied by n, and "
                                             "outputs are temperatures in both cases. The energy "
                                             "form is the regular one but the harder one to "
                                             "integrate, because n T spans 14 decades against 2 "
                                             "for T, so it needs atol around 1e-13")
    floor: float = _p(0.01, "Floor on temperatures [eV] and densities [m^-3]")
    jacobian: str = _p("numeric", "How the implicit methods get the Jacobian. numeric: by differences, "
                                  "as scipy does by default. analytic: the impurity charge-state columns "
                                  "are differentiated analytically and only the eight fluid columns are "
                                  "differenced, which removes most of the right-hand-side evaluations "
                                  "of the Jacobian. The Jacobian drives the Newton iteration and the "
                                  "step size, not the solution of the collocation equations")


@dataclass
class FHTEConfig:
    """Fast Hot Tail Estimator (runaway electrons produced by hot tail), see tqtoy.fhte."""

    enabled: bool = _p(False, "Run the FHTE on every point at the end of tqtoy run (also: tqtoy fhte results.h5)")
    n_output: int = _p(2, "Number of evaluation times from the first to the last output time "
                          "(the last one gives the final runaway fraction). With output_window = "
                          "active and only two times, the second one is the end of the runaway "
                          "window rather than the last time of the run")
    output_window: str = _p("active", "Where the evaluation times are placed. active: inside the time "
                                      "windows where E > E_c, the only ones where runaways are created "
                                      "(no backward integration outside, the runaway population is kept "
                                      "constant there). full: over the whole run, with a backward "
                                      "integration at every time")
    output_spacing: str = _p("linear", "Spacing of the evaluation times: linear or log")
    accumulate: bool = _p(True, "Keep the runaway population once it is created: the reported fraction is "
                                "the running maximum over the evaluation times. The estimator gives, at "
                                "each time, the fraction of the initial distribution that is above the "
                                "critical momentum at that time; this fraction falls when E/E_c decreases, "
                                "although the electrons that already ran away are not lost. false: report "
                                "the instantaneous fraction at each evaluation time")
    n_trajectory: int = _p(2, "Stored points per backward momentum trajectory")
    trajectory_spacing: str = _p("linear", "Spacing of the stored trajectory points: linear or log")
    field: str = _p("spitzer", "Parallel electric field: spitzer (from J with zeff and neoclassical_factor) or "
                               "model (field of the TQ model, TQ runs only)")
    zeff: float = _p(1.0, "Effective charge in the Spitzer field of the FHTE")
    neoclassical_factor: float = _p(2.0, "Factor on the Spitzer resistivity of the FHTE (neoclassical correction)")
    chandrasekhar: bool = _p(False, "Multiply the collisional drag by the Chandrasekhar function")
    temperature: str = _p("average", "Bulk temperature from a TQ run: average (density weighted), cold or hot")
    bound_electron_weight: float = _p(0.0, "Weight of the electrons bound to impurities in the collisional "
                                           "density (0: free electrons only)")
    T_floor: float = _p(10.0, "Floor on the bulk temperature", "eV")
    n_floor: float = _p(1.0e15, "Floor on the density", "m^-3")
    critical_iterations: int = _p(4, "Fixed-point iterations on gamma_c for the critical momentum")
    rtol: float = _p(1.0e-5, "Relative tolerance of the backward momentum integration")
    atol: float = _p(1.0e-10, "Absolute tolerance of the backward momentum integration")
    gamma_max: float = _p(4.0, "Upper bound of the Maxwell-Juttner integration in gamma")
    gamma_points: int = _p(200000, "Number of points of the Maxwell-Juttner integration")

    def validate(self, two_populations: bool = True) -> None:
        _choice("fhte.field", self.field, ("spitzer", "model"))
        _choice("fhte.temperature", self.temperature, ("average", "cold", "hot"))
        _choice("fhte.output_spacing", self.output_spacing, ("linear", "log"))
        _choice("fhte.output_window", self.output_window, ("active", "full"))
        _choice("fhte.trajectory_spacing", self.trajectory_spacing, ("linear", "log"))
        if self.temperature == "hot" and not two_populations:
            raise ConfigError("fhte.temperature = 'hot' requires model.populations = 2.")
        for name in ("n_output", "n_trajectory"):
            if getattr(self, name) < 2:
                raise ConfigError(f"fhte.{name} must be >= 2.")
        for name in ("zeff", "neoclassical_factor", "T_floor", "n_floor", "rtol", "atol",
                     "critical_iterations", "gamma_points"):
            if not getattr(self, name) > 0:
                raise ConfigError(f"fhte.{name} must be positive.")
        if self.gamma_max <= 1:
            raise ConfigError("fhte.gamma_max must be larger than 1.")
        if self.bound_electron_weight < 0:
            raise ConfigError("fhte.bound_electron_weight must be >= 0.")


@dataclass
class OutputConfig:
    """Output file and energy diagnostics."""

    file: str = _p("results/tqtoy_results.h5", "HDF5 output file, created if its directory does not "
                                               "exist (tqtoy run -o overrides it)")
    energy_integrals: bool = _p(True, "Integrate the energy terms on the internal solver steps")
    energy_refinement: int = _p(8, "Sub-intervals per solver step for these integrals")


@dataclass
class Config:
    """Complete simulation configuration."""

    plasma: PlasmaConfig = _section(PlasmaConfig, "Pre-disruption plasma")
    impurities: Dict[str, ImpurityConfig] = field(
        default_factory=lambda: {"neon": ImpurityConfig(injected_atom_density=5.0e19)},
        metadata={"help": "Impurity elements, one entry per element", "unit": ""})
    injection: InjectionConfig = _section(InjectionConfig, "Material injection")
    model: ModelConfig = _section(ModelConfig, "Physics options")
    time: TimeConfig = _section(TimeConfig, "Output time grid")
    solver: SolverConfig = _section(SolverConfig, "ODE solver")
    output: OutputConfig = _section(OutputConfig, "Output")
    fhte: FHTEConfig = _section(FHTEConfig, "Fast Hot Tail Estimator")
    scan: Dict[str, Any] = field(default_factory=dict, metadata={
        "help": "Scan axes: {dotted.path: values}. Values: list, {linspace: [a, b, n]}, "
                "{geomspace: [a, b, n]} or {logspace: [exp_a, exp_b, n]}", "unit": ""})

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Config":
        cfg = _build(cls, data or {}, "")
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path) -> "Config":
        data = read_yaml(path)
        try:
            return cls.from_dict(data)
        except ConfigError as exc:
            raise ConfigError(f"{path}: {exc}") from None

    @classmethod
    def from_yaml_string(cls, text: str) -> "Config":
        return cls.from_dict(parse_yaml(text))

    # ------------------------------------------------------------------ export
    def to_dict(self) -> dict:
        return _to_plain(dataclasses.asdict(self))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    # ------------------------------------------------------------------ helpers
    @property
    def element_names(self) -> List[str]:
        return list(self.impurities.keys())

    def with_overrides(self, overrides: Dict[str, Any]) -> "Config":
        """Return a copy with dotted-path parameters replaced, e.g. {'injection.n_D': 1e21}."""
        data = self.to_dict()
        for path, value in overrides.items():
            _set_path(data, path, value)
        return Config.from_dict(data)

    def scan_axes(self) -> List[Tuple[str, np.ndarray]]:
        """Scan axes as a list of (dotted path, values), in the order of the YAML file."""
        axes = []
        for path, spec in self.scan.items():
            values = parse_values(spec, path)
            if path.startswith("scan"):
                raise ConfigError("scan: a scan axis cannot point into the scan section.")
            # Check that the path exists and that the value has the right type. The
            # consistency between parameters is checked on complete scan points.
            data = dataclasses.replace(self, scan={}).to_dict()
            _set_path(data, path, _to_plain(values[0]))
            _build(Config, data, "")
            axes.append((path, values))
        return axes

    def scan_points(self):
        """Yield (index tuple, overrides dict) for every point of the scan grid."""
        axes = self.scan_axes()
        if not axes:
            yield (), {}
            return
        ranges = [range(len(v)) for _, v in axes]
        for idx in itertools.product(*ranges):
            yield idx, {p: _to_plain(v[i]) for (p, v), i in zip(axes, idx)}

    def validate(self) -> None:
        """Check the configuration. With a scan, the first scan point is checked (the
        other points are checked by run_scan before any run)."""
        if self.scan:
            axes = self.scan_axes()
            first = {path: _to_plain(values[0]) for path, values in axes}
            try:
                dataclasses.replace(self, scan={}).with_overrides(first)
            except ConfigError as exc:
                raise ConfigError(f"first scan point {first}: {exc}") from None
            return
        p, inj, m, t = self.plasma, self.injection, self.model, self.time
        if not self.impurities:
            raise ConfigError("impurities: at least one impurity element is required.")
        if m.populations not in (1, 2):
            raise ConfigError("model.populations must be 1 or 2.")
        _choice("model.resistivity", m.resistivity, ("spitzer", "multi_species"))
        _choice("model.current_sharing", m.current_sharing, ("density_weighted", "spitzer"))
        _choice("injection.source", inj.source, ("prescribed", "ablation"))
        _choice("time.spacing", t.spacing, ("log", "linear"))
        if m.resistivity == "multi_species" and m.populations == 1:
            raise ConfigError("model.resistivity = 'multi_species' requires model.populations = 2.")
        if inj.source == "ablation" and len(self.impurities) != 1:
            raise ConfigError("injection.source = 'ablation' supports a single impurity element.")
        if (p.J_final is None) != (p.J_ramp_time is None):
            raise ConfigError("plasma.J_final and plasma.J_ramp_time must be given together.")
        if inj.duration <= 0:
            raise ConfigError("injection.duration must be positive.")
        t_start = self.t_output_start
        if t.t_end <= t_start:
            raise ConfigError("time.t_end must be larger than the first output time.")
        if t.spacing == "log" and t_start <= 0:
            raise ConfigError("time.spacing = 'log' requires a positive first output time "
                              "(time.t_start or injection.t_start).")
        if t.n_output < 2:
            raise ConfigError("time.n_output must be >= 2.")
        if inj.charge_state < 0:
            raise ConfigError("injection.charge_state must be >= 0.")
        positive = {"plasma.Te0": p.Te0, "plasma.ne0": p.ne0, "plasma.cold_T0": p.cold_T0,
                    "plasma.cold_n0": p.cold_n0, "plasma.B": p.B, "solver.rtol": self.solver.rtol,
                    "solver.atol": self.solver.atol, "injection.ablation.volume": inj.ablation.volume,
                    "injection.ablation.shard_radius": inj.ablation.shard_radius,
                    "injection.ablation.n_shards": inj.ablation.n_shards,
                    "output.energy_refinement": self.output.energy_refinement,
                    "model.atomic_rate_multiplier": m.atomic_rate_multiplier,
                    "model.main_ion_mass": m.main_ion_mass,
                    "model.stochastic.coulomb_log": m.stochastic.coulomb_log,
                    "model.radiation_limit.Te_below": m.radiation_limit.Te_below,
                    "model.radiation_limit.time_scale": m.radiation_limit.time_scale,
                    "injection.ablation.switch_on_time": inj.ablation.switch_on_time,
                    "solver.floor": self.solver.floor}
        for opt in ("Ti0", "ni0", "J_ramp_time"):
            if getattr(p, opt) is not None:
                positive[f"plasma.{opt}"] = getattr(p, opt)
        for name, value in positive.items():
            if not value > 0:
                raise ConfigError(f"{name} must be positive (got {value}).")
        non_negative = {"plasma.J": p.J, "injection.n_D": inj.n_D, "injection.t_start": inj.t_start,
                        "model.linear_loss_coefficient": m.linear_loss_coefficient}
        for el, c in self.impurities.items():
            non_negative[f"impurities.{el}.injected_atom_density"] = c.injected_atom_density
            non_negative[f"impurities.{el}.background_density"] = c.background_density
        for name, value in non_negative.items():
            if value < 0:
                raise ConfigError(f"{name} must be >= 0 (got {value}).")
        _choice("solver.method", self.solver.method, SOLVER_METHODS)
        _choice("solver.jacobian", self.solver.jacobian, ("numeric", "analytic"))
        _choice("solver.state_variables", self.solver.state_variables, ("energy", "temperature"))
        if self.solver.jacobian == "analytic" and self.solver.method not in IMPLICIT_METHODS:
            raise ConfigError(f"solver.jacobian = analytic needs an implicit method "
                              f"({', '.join(IMPLICIT_METHODS)}), not {self.solver.method}.")
        self.fhte.validate(two_populations=m.populations == 2)
        st = m.stochastic
        if st.enabled and not st.r / st.a < 1 / np.sqrt(6):
            raise ConfigError(f"model.stochastic: r/a = {st.r / st.a:.3g} must be below 1/sqrt(6) = 0.408. "
                              "Above, the loss of the Te ~ (1 - r^2/a^2) profile model changes sign.")
        if st.r >= st.a:
            raise ConfigError("model.stochastic.r must be smaller than model.stochastic.a.")

    @property
    def t_output_start(self) -> float:
        return self.time.t_start if self.time.t_start is not None else self.injection.t_start

    def output_times(self) -> np.ndarray:
        """Output time grid [s]."""
        t0, t1, n = self.t_output_start, self.time.t_end, self.time.n_output
        if self.time.spacing == "log":
            return 10**np.linspace(np.log10(t0), np.log10(t1), n)
        return np.linspace(t0, t1, n)


# --------------------------------------------------------------------------- documentation

def parameter_table() -> List[Dict[str, str]]:
    """All configuration parameters: path, type, default, unit and description.

    Impurity parameters are listed as ``impurities.<element>.<name>``.
    """
    rows: List[Dict[str, str]] = []

    def type_name(tp) -> str:
        tp, optional = _unwrap_optional(tp)
        name = getattr(tp, "__name__", str(tp))
        return f"{name} or null" if optional else name

    def walk(cls, prefix: str, instance) -> None:
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            path = f"{prefix}{f.name}"
            tp = hints[f.name]
            inner, _ = _unwrap_optional(tp)
            value = getattr(instance, f.name)
            if dataclasses.is_dataclass(inner) and value is not None:
                walk(inner, path + ".", value)
                continue
            if dataclasses.is_dataclass(inner):          # optional section, unset by default
                rows.append(dict(path=path, type="section or null", default="null",
                                 unit="", help=f.metadata.get("help", "")))
                walk(inner, path + ".", inner())
                continue
            if path == "impurities":
                walk(ImpurityConfig, "impurities.<element>.", ImpurityConfig())
                continue
            if path == "scan":
                rows.append(dict(path=path, type="mapping", default="{}", unit="", help=f.metadata["help"]))
                continue
            if value is None:
                default = "null"
            elif isinstance(value, bool):
                default = str(value).lower()
            elif isinstance(value, float):
                default = f"{value:g}"
            else:
                default = str(value)
            rows.append(dict(path=path, type=type_name(tp), default=default,
                             unit=f.metadata.get("unit", ""), help=f.metadata.get("help", "")))

    walk(Config, "", Config())
    return rows


def parameter_unit(path: str) -> str:
    """Unit of a configuration parameter given by its dotted path ('' if none or unknown)."""
    parts = path.split(".")
    if parts[0] == "impurities" and len(parts) == 3:
        path = f"impurities.<element>.{parts[2]}"
    for row in parameter_table():
        if row["path"] == path:
            return row["unit"]
    return ""


# --------------------------------------------------------------------------- value specs

def parse_values(spec: Any, path: str = "") -> np.ndarray:
    """Convert a scan specification into a 1D array.

    Accepted forms: a scalar, a list of values, or a mapping with one key among
    ``linspace: [start, stop, num]``, ``logspace: [exp_start, exp_stop, num]``
    (numpy convention, base 10) and ``geomspace: [start, stop, num]``.
    """
    if isinstance(spec, dict):
        if len(spec) != 1:
            raise ConfigError(f"scan.{path}: use exactly one of linspace/logspace/geomspace.")
        (kind, args), = spec.items()
        funcs = {"linspace": np.linspace, "logspace": np.logspace, "geomspace": np.geomspace}
        if kind not in funcs:
            raise ConfigError(f"scan.{path}: unknown generator '{kind}' (use {', '.join(funcs)}).")
        if not isinstance(args, (list, tuple)) or len(args) != 3:
            raise ConfigError(f"scan.{path}: {kind} needs [start, stop, num].")
        start, stop, num = _as_float(args[0], path), _as_float(args[1], path), int(args[2])
        return funcs[kind](start, stop, num)
    if isinstance(spec, (list, tuple, np.ndarray)):
        return np.array([_as_scalar(v, path) for v in spec])
    return np.array([_as_scalar(spec, path)])


# --------------------------------------------------------------------------- internals

def _choice(name, value, allowed):
    if value not in allowed:
        raise ConfigError(f"{name} = '{value}' is invalid (allowed: {', '.join(allowed)}).")


def _as_float(v, path):
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ConfigError(f"{path}: expected a number, got {v!r}.") from None


def _as_scalar(v, path):
    if isinstance(v, (bool, int, float, np.number)):
        return v
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    raise ConfigError(f"{path}: invalid value {v!r}.")


def _unwrap_optional(tp):
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _coerce(tp, value, path):
    """Coerce a YAML value to the annotated type ``tp``."""
    tp, optional = _unwrap_optional(tp)
    if tp is Any:
        return value
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{path}: a value is required.")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping.")
        return _build(tp, value, path)
    origin = typing.get_origin(tp)
    if origin in (dict, Dict):
        key_tp, val_tp = typing.get_args(tp)
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping.")
        out = {}
        for k, v in value.items():
            if v is None and dataclasses.is_dataclass(val_tp):
                v = {}                   # e.g. "neon:" with no field -> defaults
            out[str(k)] = _coerce(val_tp, v, f"{path}.{k}")
        return out
    if origin in (list, List):
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {value!r}.")
        (item_tp,) = typing.get_args(tp) or (Any,)
        return [_coerce(item_tp, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected true or false, got {value!r}.")
    if tp is int:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected an integer, got {value!r}.")
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{path}: expected an integer, got {value!r}.") from None
        if f != int(f):
            raise ConfigError(f"{path}: expected an integer, got {value!r}.")
        return int(f)
    if tp is float:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected a number, got {value!r}.")
        return _as_float(value, path)  # also accepts YAML strings such as '1e20'
    if tp is str:
        return str(value)
    return value


def _build(cls, data: dict, path: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'configuration'}: expected a mapping.")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = [k for k in data if k not in names]
    if unknown:
        where = path or "top level"
        raise ConfigError(f"{where}: unknown key(s) {unknown}. Valid keys: {names}.")
    kwargs = {}
    for name in names:
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], f"{path}.{name}" if path else name)
    return cls(**kwargs)


def _to_plain(obj):
    """Convert numpy scalars/arrays to plain Python types (YAML/HDF5 friendly)."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _set_path(data: dict, path: str, value) -> None:
    """Set a dotted-path value in a nested dict. New impurity elements may be created."""
    keys = path.split(".")
    if keys[0] == "scan" and len(keys) > 1:      # scan axes are keyed by dotted paths
        data.setdefault("scan", {})[".".join(keys[1:])] = _to_plain(value)
        return
    node = data
    created = False
    for depth, key in enumerate(keys[:-1]):
        if isinstance(node, dict) and key in node:
            if node[key] is None:        # optional section not set yet (e.g. ablation.parks)
                node[key] = {}
                created = True
            node = node[key]
        elif isinstance(node, dict) and depth == 1 and keys[0] == "impurities":
            node[key] = {}               # new impurity element
            created = True
            node = node[key]
        else:
            raise ConfigError(f"'{path}' is not a configuration parameter.")
    last = keys[-1]
    # Unknown field names inside a newly created section are caught by from_dict().
    if not isinstance(node, dict) or (last not in node and not created):
        raise ConfigError(f"'{path}' is not a configuration parameter.")
    node[last] = _to_plain(value)


def parse_assignment(text: str) -> Tuple[str, Any]:
    """Parse 'dotted.path=value' (value in YAML syntax) from the command line."""
    if "=" not in text:
        raise ConfigError(f"'{text}': expected key=value.")
    key, raw = text.split("=", 1)
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        value = _as_scalar(value, key)
    return key.strip(), value

