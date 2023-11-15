import numpy as np

def inclination(pars):
    """
    Calculates the inclination of the orbit.

    Parameters:
    -----------
    pars : dict
        A dictionary containing the following keys:
        - e: eccentricity of the orbit
        - w: argument of periastron (in degrees)
        - b: impact parameter
        - aR: semi-major axis of the orbit in units of the stellar radius

    Returns:
    --------
    inclination : float
        The inclination of the orbit in degrees.
    """
    #if e and w not in dictionary, assume circular orbit
    if "e" not in pars.keys():
        pars["e"] = 0
    if "w" not in pars.keys():
        pars["w"] = 90
    ecc_fac = (1-pars["e"]**2)/(1 + pars["e"]*np.sin(np.deg2rad(pars["w"])))
    return np.arccos(pars["b"]/(pars["aR"]*ecc_fac)) * 180/np.pi

def impact_parameter(pars):
    """
    Calculates the impact parameter of a planet transit.

    Parameters:
    -----------
    pars : dict
        A dictionary containing the following keys:
        - aR : float
            The ratio of the planet's radius to the star's radius.
        - inc : float
            The inclination of the planet's orbit in degrees.
        - e : float
            The eccentricity of the planet's orbit.
        - w : float
            The argument of periastron in degrees.

    Returns:
    --------
    float
        The impact parameter of the planet transit.
    """
    #if e and w not in dictionary, assume circular orbit
    if "e" not in pars.keys():
        pars["e"] = 0
    if "w" not in pars.keys():
        pars["w"] = 90
    ecc_fac = (1-pars["e"]**2)/(1 + pars["e"]*np.sin(np.deg2rad(pars["w"])))
    return pars["aR"]*np.cos(np.deg2rad(pars["inc"]))*ecc_fac

def transit_duration(pars):
    """
    Calculates the transit duration of a planet orbiting a star.

    Parameters:
    -----------
    pars : dict
        A dictionary containing the following keys:
        - "P": orbital period of the planet in days
        - "e": eccentricity of the planet's orbit
        - "w": longitude of periastron in degrees
        - "inc": inclination of the planet's orbit in degrees
        - "aR": semi-major axis of the planet's orbit in units of the stellar radius
        - "Rv": ratio of the planet's radius to the stellar radius

    Returns:
    --------
    tdur : float
        The transit duration of the planet in the same units as `pars["P"]`.
    """
    ecc_fac = (1-pars["e"]**2)/(1 + pars["e"]*np.sin(np.deg2rad(pars["w"])))
    sini       = np.sin(np.deg2rad(pars["inc"]))
    cosi       = np.cos(np.deg2rad(pars["inc"]))

    denom      = pars["aR"]*ecc_fac*sini

    tdur       =  (pars["P"]/np.pi) * (ecc_fac**2/np.sqrt(1-pars["e"]**2)) * (np.arcsin( np.sqrt(1+pars["Rv"]**2 - (pars["aR"]*ecc_fac*cosi)**2 )/denom ))
    return tdur

def phase_fold(t, pars):
    """
    Phase folds a given time series using the provided parameters.

    Parameters:
    t (array-like): The time series to be phase folded.
    pars (dict): A dictionary containing the parameters for phase folding. Must contain the keys "t0" and "P".

    Returns:
    array-like: The phase folded time series.
    """
    phase = (t-pars["t0"])/pars["P"] % 1    
    return phase


def convert_radius(R_pl, hf, qmass, inc, aR, u1=0, u2=0, conv = "Rp2Rv", plot=False):
    """
        convert between spherical planet radius Rp and ellipsodial volumetric radius Rv.
        This is done by finding the best fit Rp (or Rv) to the simulated deformed (or spherical) planet light + \
        curve with other parameters kept the same. 

        Parameters:
        -----------
        R_pl: radius of planet to convert (units of stellar radii)
        hf  : Love number of the deformed planet
        qmass: Mass ratio Mp/Mstar
        inc: inclination in degrees
        aR : scale semi-major axis a/Rstar
        u1,u2: (optional) quadratic limb darkening parameters
        conv : required radius conversion. "Rp2Rv" to convert from spherical to deformed and "Rv2Rp" for the reverse.
        plot : bool, True to plot fit.  

        Returns:
        ---------
        R_fit: best fit radius (of deformed or spherical planet) that produces similar depth as the inputted planet radius

    """
    import ellc
    from matplotlib import pyplot as plt
    r_1 = 1./aR            #Rst/a
    r_2 = R_pl/aR            # Rp/a
    b = aR*np.cos(np.deg2rad(inc))
    tdur = np.arcsin(r_1*np.sqrt( ((1+r_2)**2-b**2) / (1-b**2*r_1**2) ))/np.pi
    ph =  np.linspace(-tdur, tdur,500)
    
    assert conv in ["Rp2Rv", "Rv2Rp"],f'conv must be one of ["Rp2Rv", "Rv2Rp"] but {conv} given'
    pl_shape = "sphere" if conv=="Rp2Rv" else "love"
    
    def model(R_pl,pl_shape=pl_shape,data=None):
    
        ellc_flux = ellc.lc(ph, t_zero=0, radius_1=r_1,radius_2=R_pl*r_1,
                            incl=inc, sbratio=0,
                            ld_1="quad", ldc_1=[u1,u2], 
                            shape_2=pl_shape, q=qmass, hf_2=hf)
        
        if data is None:
            return ellc_flux
        return np.sum((data - ellc_flux)**2)
    
    flux_sim = model(R_pl,pl_shape)
    rprint = "Rp" if pl_shape=="sphere" else "Rv"
    
    if plot: 
        fig,ax = plt.subplots(2,1, figsize=(7,5), gridspec_kw={"height_ratios":(3,1)}, sharex=True)
        ax[0].plot(ph, flux_sim, "b",lw=3,label=f"sim – {pl_shape} ({rprint} = {R_pl:.4f})")
    
    #fit simulated light curve
    from scipy.optimize import minimize_scalar, differential_evolution
    fit_shape = "love" if pl_shape=="sphere" else "sphere"
    
    res =  minimize_scalar(model,(0.8*R_pl, R_pl, 1.2*R_pl),args=(fit_shape,flux_sim))
    R_fit = res.x
    
    flux_fit = model(R_fit,fit_shape)
    rprint = "Rp" if fit_shape=="sphere" else "Rv"
    
    if plot:
        ax[0].plot(ph, flux_fit,"r--", lw=2, label = f"fit – {fit_shape} ({rprint} = {R_fit:.4f})"); ax[0].legend()
        ax[0].set_ylabel("Flux")
        ax[1].plot(ph, 1e6*(flux_sim-flux_fit))
        ax[1].set_xlabel("Phase")
        ax[1].set_ylabel("res [ppm]")
        plt.subplots_adjust(hspace=0.01)
        
    return R_fit