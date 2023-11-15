import ellc
import batman
import numpy as np
from matplotlib import pyplot as plt
from .utils import transit_duration,phase_fold


class deformed_PC:
    def __init__(self, deformed=True, ld_law="power-2", planet_variation="cosine", custom_planet_variation_params=None, ellc_grid="default"):
        """
        Initializes a deformed planet object with optional ellipsoidal variation and Doppler beaming signals.

        Args:
        - deformed (bool): If True, the planet is assumed to be deformed.
        - ld_law (str): The limb darkening law to use. Can be "power-2" or "quad".
        - planet_variation (str): The type of planet variation to use. Can be "cosine" or a custom function. #TODO add more options like spiderman model
        - custom_planet_variation_params (dict): A dictionary of parameters for the custom planet variation function. Only used if planet_variation is a custom function.
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

        pars = {"t0":0, "P":1, "Rv":0.1, "aR":3, "inc":90, "e":0, "w":90, "ld_pars":[0.5,0.5],"qmass":1, "Fp":0, "Fn":0, "hf":1.5, "delta":0, "A_EV":0, "A_DB":0}
        if not self.deformed: 
            pars["qmass"] = 1
            pars["hf"] = 0

        #TODO: find a way to specify which parameters to vary and which to keep fixed
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

        delta_rad  = np.deg2rad(pars["delta"])   #phase-offset
        atm_signal =  (pars["Fp"]-pars["Fn"]) * (1- np.cos( self.phi + delta_rad))/2 + pars["Fn"]
        return atm_signal*self.ppm
    
    def planet_lambertian_variation(self,t, pars):
        """
        Calculates the Lambertian function for the planet's atmospheric phase variation.

        Args:
        - t (array): An array of times at which to calculate the Lambertian function.
        - pars (dict): A dictionary of parameters for the planet.
        
        Returns:
        - delta (array): An array of the Lambertian function values for the planet's atmospheric phase variation.
        """

        delta_rad  = np.deg2rad(pars["delta"])
        NotImplementedError("Lambertian variation not yet implemented")
        return atm_signal*self.ppm
        


    def projected_planet_area(self,pars,phase=None,return_axis=False):
        """
        Calculates the projected area of the ellipsoidal planet as a function of phase. From  eqn B.9 of leconte 2011(http://www.aanda.org/10.1051/0004-6361/201015811e)

        Args:
        - pars (dict): A dictionary containing the following keys:
            - "hf" (float): The fractional radius of the planet's Hill sphere.
            - "qmass" (float): The planet-to-star mass ratio.
            - "Rv" (float): The planet's radius.
            - "aR" (float): The planet's semi-major axis in units of the star's radius.
            - "inc" (float): The planet's inclination in degrees.
        - return_axis (bool): If True, returns the semi-major, semi-minor, and semi-intermediate axes of the ellipsoid.

        Returns:
        - L (float): The projected area of the ellipsoidal planet as a function of phase for hf=0, L is constant with phase.
        - abc (tuple): A tuple containing the semi-major, semi-minor, and semi-intermediate axes of the ellipsoid. Only returned if `return_axis` is True.
        """
        qr = 0.5*pars["hf"]*1/pars["qmass"]*(pars["Rv"]/pars["aR"])**3
        bx = pars["Rv"]*(1-(2./3.)*qr + (17./9.)*qr**2 - (328./81.)*qr**3 + (2558./243.)*qr**4)
        ax = bx*(1.+3.*qr)          #eqn(10) correia 2014
        cx = bx*(1.-qr)
        abc=np.array([ax,bx,cx])

        phi = self.phi if hasattr(self,'phi') else 2*np.pi*phase 

        a1,a2,a3 = abc
        d2r = np.deg2rad
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        sininc = np.sin(d2r(pars["inc"]))
        cosinc = np.cos(d2r(pars["inc"]))
        L = np.pi*np.sqrt(a3**2*sininc**2*(a1**2*sinphi**2+a2**2*cosphi**2) + a1**2*a2**2*cosinc**2)
        return L if not return_axis else (L, abc)


    def ellipsoidal_variation(self,t, pars):
        """
        Calculates the ellipsoidal variation signal for a given time and set of parameters.

        Args:
        t (float): The time at which to calculate the signal.
        pars (dict): A dictionary containing the EV semi-amplitude (pars["A_EV"]) in ppm
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
        # m1 = batman.TransitModel(params, t)    #initializes model
        # params.t_secondary = m1.get_t_secondary(params)
        m2 = batman.TransitModel(params, t, transittype="secondary")
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

            self.phase = phase_fold(t,pars)
            phi      = 2*np.pi*self.phase
            self.phi = phi 

            trans_signal     = self.transit_signal(t, pars)
            ecl_signal       = self.eclipse_signal(t, pars)
            atm_signal       = self.atm_phase_var(t, pars)
            ellip_signal     = self.ellipsoidal_variation(t, pars)
            DB_signal        = self.doppler_beaming(t, pars)

            ellipsoidal_area = self.projected_planet_area(pars) if self.deformed else np.ones_like(self.phase)
            normalized_area  = ellipsoidal_area/ellipsoidal_area.min()

            if return_components:
                def_atm_signal = atm_signal * normalized_area
                pc_def_contrib = def_atm_signal - atm_signal
                return def_atm_signal, ellip_signal, DB_signal, pc_def_contrib

            tdur  = transit_duration(pars)/pars["P"]
            tmask = np.abs(self.phase) <= tdur/2
            normalized_area[tmask] = 1

            def_atm_signal_mod = atm_signal * normalized_area

            star_varying_signal = 1 + ellip_signal + DB_signal
            pc = (trans_signal * star_varying_signal) + (ecl_signal * def_atm_signal_mod)
            return pc




