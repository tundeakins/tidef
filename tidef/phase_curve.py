import ellc
import batman
import numpy as np
from matplotlib import pyplot as plt
  
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


class deformed_PC:
    def __init__(self, deformed=True, ld_law="power-2", planet_variation="cosine", ellc_grid="default"):
        """
        Initializes a deformed planet object with optional ellipsoidal variation and Doppler beaming signals.

        Args:
        - deformed (bool): If True, the planet is assumed to be deformed.
        - ld_law (str): The limb darkening law to use. Can be "power-2" or "quad".
        - planet_variation (str): The type of planet variation to use. Can be "cosine" or a function. #TODO add more options like spiderman model
        - ellc_grid (str): The grid resolution to use for ellc. Can be "default", "sparse", or "fine".
        
        Returns:
        - None
        """
        self.deformed  = deformed
        self.ppm       = 1e-6
        self.ellc_grid = ellc_grid
        self.pars      = self._default_pars()
        self.LDC_model = ld_law

        self.atm_phase_var  = self.planet_cosine_variation if planet_variation=="cosine" else planet_variation
        assert self.LDC_model in ["power-2", "quad"], "ld_law not supported. must be one of ['power-2', 'quad']"

    def _default_pars(self):
        """
        Returns a dictionary of default parameters for the planet.

        Args:
        - None
        
        Returns:
        - pars (dict): A dictionary of default parameters for the planet.
        """
        pars = {"t0":0, "P":1, "Rv":0.1, "aR":3, "inc":90, "e":0, "w":90, "ld_pars":[0.5,0.5],"qmass":0.00098, "Fp":1000, "Fn":200, "hf":1.5, "delta":0, "A_EV":0, "A_DB":0}
        if not self.deformed: 
            pars["qmass"] = 1
            pars["hf"] = 0
        return pars

    def planet_cosine_variation(self,t, pars):
        """
        Calculates the cosine function for the planet's atmospheric phase variation.

        Args:
        - t (array): An array of times at which to calculate the cosine function.
        - pars (dict): A dictionary of parameters for the planet.
        
        Returns:
        - delta (array): An array of the cosine function values for the planet's atmospheric phase variation.
        """

        delta_rad = np.deg2rad(pars["delta"])   #phase-offset
        atm_signal =  (pars["Fp"]-pars["Fn"]) * (1- np.cos( self.phi + delta_rad))/2 + pars["Fn"]
        return atm_signal*self.ppm


    def projected_planet_area(self,pars,return_axis=False):
        """
        Calculates the projected area of the ellipsoidal planet as a function of phase.

        Args:
        - pars (dict): A dictionary containing the following keys:
            - "hf" (float): The fractional radius of the planet's Hill sphere.
            - "qmass" (float): The planet-to-star mass ratio.
            - "Rv" (float): The planet's radius.
            - "aR" (float): The planet's semi-major axis in units of the star's radius.
            - "inc" (float): The planet's inclination in degrees.
        - return_axis (bool): If True, returns the semi-major, semi-minor, and semi-intermediate axes of the ellipsoid.

        Returns:
        - L (float): The projected area of the ellipsoidal planet as a function of phase.
        - abc (tuple): A tuple containing the semi-major, semi-minor, and semi-intermediate axes of the ellipsoid. Only returned if `return_axis` is True.
        """
        qr = 0.5*pars["hf"]*1/pars["qmass"]*(pars["Rv"]/pars["aR"])**3
        bx = pars["Rv"]*(1-(2./3.)*qr + (17./9.)*qr**2 - (328./81.)*qr**3 + (2558./243.)*qr**4)
        ax = bx*(1.+3.*qr)          #eqn(10) correia 2014
        cx = bx*(1.-qr)
        abc=np.array([ax,bx,cx])

        a1,a2,a3 = abc
        d2r = np.deg2rad
        cosphi = np.cos(self.phi)
        sinphi = np.sin(self.phi)
        sininc = np.sin(d2r(pars["inc"]))
        cosinc = np.cos(d2r(pars["inc"]))
        L = np.pi*np.sqrt(a3**2*sininc**2*(a1**2*sinphi**2+a2**2*cosphi**2) + a1**2*a2**2*cosinc**2)
        return L if not return_axis else (L, abc)


    def ellipsoidal_variation(self,t, pars):
        """
        Calculates the ellipsoidal variation signal for a given time and set of parameters.

        Args:
        t (float): The time at which to calculate the signal.
        pars (dict): A dictionary containing the parameters (pars["A_EV"])needed to calculate the signal.

        Returns:
        float: The calculated ellipsoidal variation signal, in parts per million (ppm).
        """
        #ellipsoidal variation        
        ellip_signal = pars["A_EV"]*(1- np.cos(2*self.phi))
        return ellip_signal*self.ppm

    def doppler_beaming(self,t, pars):
        """
        Calculates the Doppler beaming signal for a given time and set of parameters.
        
        """
        #doppler beaming
        DB_signal = pars["A_DB"]*(np.sin(self.phi))
        return DB_signal*self.ppm

    def transit_signal(self, t, pars):
        """
        Calculates the transit signal for a given set of parameters.

        Args:
            t (array-like): Array of time values.
            pars (dict): Dictionary of transit parameters.

        Returns:
            array-like: Array of transit signal values.
        """
        if self.deformed:
            fc = np.sqrt(pars["e"])*np.cos(np.deg2rad(pars["w"]))
            fs = np.sqrt(pars["e"])*np.sin(np.deg2rad(pars["w"]))
            # print("using ellc for ellipsoidal planet transit")
            trans_signal =  ellc.lc(t, t_zero=pars["t0"], period=pars["P"], radius_1=1/pars["aR"],
                                radius_2=pars["Rv"]/pars["aR"], incl=pars["inc"],  sbratio=0, f_c =fc, f_s=fs,
                                ld_1=self.LDC_model, ldc_1=pars["ld_pars"], shape_1='sphere', shape_2="love", 
                                grid_1=self.ellc_grid, grid_2=self.ellc_grid,q=pars["qmass"], hf_2=pars["hf"])
        else:
            # print("using batman for spherical planet transit")
            params = self._PCpars_to_batman(pars)
            m1 = batman.TransitModel(params, t)    #initializes model
            trans_signal = m1.light_curve(params)                    #calculates transit

        return trans_signal

    def eclipse_signal(self,t,pars):
        """
        Calculates the eclipse signal for a given set of parameters and time array.

        Args:
        t (array-like): Time array.
        pars (dict): Dictionary of parameters.

        Returns:
        array-like: Eclipse signal rescaled to 0-1.
        """
        params = self._PCpars_to_batman(pars)
        m1 = batman.TransitModel(params, t)    #initializes model
        params.t_secondary = m1.get_t_secondary(params)
        m2 = batman.TransitModel(params, t, transittype="secondary")
        # print("tsec3", m2.get_t_secondary(params))
        ecl_signal = self._rescale(m2.light_curve(params)) #ECLIPSE rescaled to 0-1
        return ecl_signal


    def _rescale(self,signal):
        #rescales signal to 0-1
        return (signal-min(signal))/np.ptp(signal) if np.all(min(signal) != max(signal)) else signal
    
    def _PCpars_to_batman(self,pars):
        #converts phase curve parameters to batman parameters
        params             = batman.TransitParams()       #object to store batman transit parameters
        params.t0          = pars["t0"]                #time of inferior conjunction
        params.per         = pars["P"]                #orbital period
        params.rp          = pars["Rv"]                #planet radius (in units of stellar radii)
        params.a           = pars["aR"]                 #semi-major axis (in units of stellar radii)
        params.inc         = pars["inc"]              #orbital inclination (in degrees)
        params.ecc         = pars["e"]            #eccentricity
        params.w           = pars["w"]                #longitude of periastron (in degrees)
        params.limb_dark   = "power2" if self.LDC_model ==  "power-2" else "quadratic"  if self.LDC_model=="quad" else self.LDC_model     #limb darkening model
        params.u           = pars["ld_pars"]          #limb darkening coefficients [u1, u2, u3, u4]
        params.fp          = pars["Fp"] *self.ppm               #planet flux
        params.t_secondary = params.t0 + 0.5*params.per*(1+4/np.pi*params.ecc * np.cos(np.deg2rad(params.w)))   #time of secondary eclipse 

        return params



    def _inclination(self,pars):
        #inclination of the orbit 
        ecc_fac = (1-pars["e"]**2)/(1 + pars["e"]*np.sin(np.deg2rad(pars["w"])))
        return np.arccos(pars["b"]/(pars["aR"]*ecc_fac)) * 180/np.pi

    def _impact_parameter(self,pars):
        #impact parameter
        ecc_fac = (1-pars["e"]**2)/(1 + pars["e"]*np.sin(np.deg2rad(pars["w"])))
        return pars["aR"]*np.cos(np.deg2rad(pars["inc"]))*ecc_fac

    def _transit_duration(self,pars):
        #transit duration  in same units as pars["P"]
        #eqn 30 and 31 of Kipping 2010 https://doi.org/10.1111/j.1365-2966.2010.16894.x
        ecc_fac = (1-pars["e"]**2)/(1 + pars["e"]*np.sin(np.deg2rad(pars["w"])))
        sini       = np.sin(np.deg2rad(pars["inc"]))
        cosi       = np.cos(np.deg2rad(pars["inc"]))

        denom      = pars["aR"]*ecc_fac*sini

        tdur       =  (pars["P"]/np.pi) * (ecc_fac**2/np.sqrt(1-pars["e"]**2)) * (np.arcsin( np.sqrt(1+pars["Rv"]**2 - (pars["aR"]*ecc_fac*cosi)**2 )/denom ))
        return tdur

    def _phase_fold(self, t, pars):
        #phase folding
        phase = (t-pars["t0"])/pars["P"] % 1
        return phase   
    

    def phase_curve(self, t, pars, return_components=False):
            """
            Calculates the phase curve of a planet given a set of parameters.

            Args:
                t (array-like): Array of time values.
                pars (dict): Dictionary of parameters.
                return_flux (bool, optional): Whether to return the flux. Defaults to False.

            Returns:
                array-like: Array of phase curve values.
            """
            _pars = self.pars.copy()
            _pars.update(pars)
            self.phase = self._phase_fold(t,pars)
            phi      = 2*np.pi*self.phase
            self.phi = phi   

            trans_signal     = self.transit_signal(t, pars)
            ecl_signal       = self.eclipse_signal(t, pars)
            atm_signal       = self.atm_phase_var(t, pars)
            ellip_signal     = self.ellipsoidal_variation(t, pars)
            DB_signal        = self.doppler_beaming(t, pars)

            ellipsoidal_area = self.projected_planet_area(pars)
            normalized_area  = ellipsoidal_area/ellipsoidal_area.min()

            if return_components:
                def_atm_signal = atm_signal * normalized_area
                pc_def_contrib = def_atm_signal - atm_signal
                return def_atm_signal, ellip_signal, DB_signal, pc_def_contrib

            tdur  = self._transit_duration(pars)/pars["P"]
            tmask = np.abs(self.phase) <= tdur/2
            normalized_area[tmask] = 1

            def_atm_signal_mod = atm_signal * normalized_area

            star_varying_signal = 1 + ellip_signal + DB_signal
            pc = (trans_signal * star_varying_signal) + (ecl_signal * def_atm_signal_mod)
            return pc




