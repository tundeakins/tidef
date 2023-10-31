from tidef.phase_curve import deformed_PC, convert_radius

def test_convert_radius():
    # Test conversion from spherical to deformed radius
    R_pl = 1.0
    hf = 0.1
    qmass = 0.01
    inc = 90.0
    aR = 10.0
    conv = "Rp2Rv"
    R_fit = convert_radius(R_pl, hf, qmass, inc, aR, conv=conv)
    assert np.isclose(R_fit, 1.011, rtol=1e-3)

    # Test conversion from deformed to spherical radius
    R_pl = 1.011
    hf = 0.1
    qmass = 0.01
    inc = 90.0
    aR = 10.0
    conv = "Rv2Rp"
    R_fit = convert_radius(R_pl, hf, qmass, inc, aR, conv=conv)
    assert np.isclose(R_fit, 1.0, rtol=1e-3)