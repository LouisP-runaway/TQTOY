"""Non-regression of the model against pinned reference values (needs the OpenADAS data).

Five configurations cover one and two populations, the prescribed and ablation sources, neon
and argon, and the Spitzer and multi-species resistivities. They are described in
tests/data/reference_cases.json, and two kinds of reference are stored for each:

* rhs_<case>.npz: 80 states and the right-hand side evaluated on them. The right-hand side
  involves no linear algebra, so it must be reproduced to round-off on any machine (strict
  test).
* reference_<case>.npz: the trajectory over 200 output times. The implicit solver calls LAPACK
  kernels whose round-off depends on the machine, and the loose default tolerance
  (rtol = 1e-2) amplifies that difference. Trajectories have been observed to move by up to
  2e-5 (fluid) and 1e-3 (charge states) of their peak values between two CPUs, so the
  trajectory test uses tolerances above that level.

A failure here means the physics has changed. That is legitimate when it is intended, and the
reference files are then regenerated on purpose and the change written down in the changelog.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_adas
from tqtoy import Config, simulate
from tqtoy.model import TQModel

DATA = Path(__file__).parent / "data"
SPEC = json.loads((DATA / "reference_cases.json").read_text())
CASES = sorted(SPEC["cases"])
KEYS = ["Te_hot", "Te_cold", "ne_hot", "ne_cold", "Ti_hot", "Ti_cold", "ni_hot", "ni_cold"]


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _config(case):
    spec = dict(SPEC["cases"][case])
    spec.pop("description")
    return Config.from_dict(_merge(SPEC["common"], spec))


@requires_adas
@pytest.mark.adas
@pytest.mark.parametrize("case", CASES)
def test_the_right_hand_side_matches_its_reference(case):
    ref = np.load(DATA / f"rhs_{case}.npz")
    model = TQModel(_config(case))
    for t, y, dydt in zip(ref["t"], ref["y"], ref["dydt"]):
        new = np.asarray(model.rhs(t, y), dtype=float)
        np.testing.assert_allclose(new, dydt, rtol=1e-12, atol=0.0)


@requires_adas
@pytest.mark.adas
@pytest.mark.parametrize("case", CASES)
def test_the_trajectory_matches_its_reference(case):
    ref = np.load(DATA / f"reference_{case}.npz")
    r = simulate(_config(case), t_eval=ref["t_output"])
    np.testing.assert_array_equal(r.t, ref["t"])
    for k in KEYS:
        scale = np.max(np.abs(ref[k]))
        assert np.max(np.abs(r.fluid[k] - ref[k])) <= 1e-3 * scale, k
    scale = np.max(np.abs(ref["nij"]))
    assert np.max(np.abs(r.nij - ref["nij"])) <= 1e-2 * scale
