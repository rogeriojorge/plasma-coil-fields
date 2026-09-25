"""Study-side checks: the VMEX pressure-family deck and the circular-tokamak reference."""

import sys
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest
import vmex as vj
from vmex.core.profiles import current, pressure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drivers"))
from nearaxis_finite_beta_helpers import pressure_family_input as make_input


def test_vmex_pressure_family_round_trips_signed_flux_and_zero_current():
    base = vj.VmecInput(
        phiedge=-9.966676405986645e-4,
        am=np.zeros(21),
        ac=np.zeros(21),
        ns_array=(5,),
        ftol_array=(1e-8,),
        niter_array=(20,),
    )
    inp = make_input(base, 0.02, 0.5, -6e5, nzeta=32)
    sample = np.linspace(0, 1, 19)
    np.testing.assert_allclose(inp.phiedge, base.phiedge, rtol=0, atol=0)
    assert inp.lfreeb and inp.ncurr == 1 and inp.curtor == 0
    assert inp.pres_scale == pytest.approx(120.0)
    np.testing.assert_allclose(
        current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, sample),
        0,
        atol=0,
    )
    np.testing.assert_allclose(
        pressure(
            inp.pmass_type,
            inp.am,
            inp.am_aux_s,
            inp.am_aux_f,
            sample,
            pres_scale=inp.pres_scale,
            bloat=inp.bloat,
            spres_ped=inp.spres_ped,
        ),
        120.0 * (1 - sample),
        atol=1e-12,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.runtime"
        inp.to_indata(path)
        reread = vj.VmecInput.from_file(path)
    assert reread.phiedge == base.phiedge
    assert reread.lfreeb and reread.ncurr == 1 and reread.curtor == 0
    assert reread.pres_scale == inp.pres_scale
    assert reread.mgrid_file == "essos_coils(direct)"
    np.testing.assert_allclose(reread.am[:2], [1, -1], rtol=0, atol=0)
    np.testing.assert_allclose(reread.ac, 0, atol=0)


def test_fixed_boundary_circular_tokamak_reference_coefficients():
    """Keep the finite-current fixed-boundary limit separate from the vacuum solve."""
    beta_p, minor_radius, major_radius, B0 = 0.37, 0.08, 1.2, 1.7
    X2c = (beta_p + 0.75) / major_radius
    Y2s = X2c
    B2c = -B0 * (beta_p + 0.25) / (2 * major_radius**2)
    shift_from_B2c = -minor_radius**2 * major_radius * B2c / B0
    shift_reference = minor_radius**2 * (beta_p + 0.25) / (2 * major_radius)
    assert Y2s == X2c
    np.testing.assert_allclose(shift_from_B2c, shift_reference, rtol=2e-15)


