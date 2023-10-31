from tidef.phase_curve import deformed_PC, convert_radius
import numpy as np
import pytest

def test_convert_radius():
    # Test conversion from spherical to deformed radius
    Rp = 0.1
    hf = 1.5
    qmass = 0.00098
    inc = 90.0
    aR = 3.0
    conv = "Rp2Rv"
    R_fit = convert_radius(Rp, hf, qmass, inc, aR, conv=conv)
    assert np.isclose(R_fit, 0.10342083, rtol=1e-3)

    # Test conversion from deformed to spherical radius
    Rv = 0.1
    hf = 1.5
    qmass = 0.00098
    inc = 90.0
    aR = 3.0
    conv = "Rv2Rp"
    R_fit = convert_radius(Rv, hf, qmass, inc, aR, conv=conv)
    assert np.isclose(R_fit, 0.096992882, rtol=1e-3)