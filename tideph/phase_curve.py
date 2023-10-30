import ellc
import batman
import numpy as np




class deformed_PC:
    def __init__(self, deformed=True, EV=False, DB=False, ellc_grid="default",ld_law="power-2"):
        self.EV        = EV
        self.DB        = DB
        self.deformed  = deformed
        self.ppm       = 1e-6
        self.ellc_grid = ellc_grid
        self.LDC_model = ld_law
        self.pars      = self._default_pars()

    def _default_pars(self):
        #default parameters for the planet
        pars = {"t0":0, "P":1, "rp":0.1, "aR":3, "inc":90, "e":0, "w":90, "LDC_model":"quadratic", "ld_pars":[0.5,0.5],"qmass":0.00098, "Fp":1000, "Fn":200, "qmass":1, "hf":1.5, "delta":0, "A_EV":0, "A_DB":0}
        pars["Rv"] = pars["rp"]
        return pars

    def planet_cosine_variation(self,t, pars):
        #cosine function for planet's atmospheric phase variation
        delta_rad = np.deg2rad(pars["delta"])   #phase-offset
        atm_signal =  (pars["Fp"]-pars["Fn"]) * (1- np.cos( self.phi + delta_rad))/2 + pars["Fn"]
        return atm_signal*self.ppm


    def projected_ellipse_area(self,pars,return_axis=False):
        #projected area of the ellipsoidal planet as a function of phase

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
        #ellipsoidal variation of the star
        ellip_signal = pars["A_EV"] *self.ppm *(1- np.cos(2*self.phi))
        return ellip_signal

    def doppler_beaming(self,t, pars):
        #doppler beaming
        DB_signal = pars["A_DB"] *self.ppm *(np.sin(self.phi))
        return DB_signal

    def transit_signal(self,t, pars):
        if self.deformed:
            # print("using ellc for ellipsoidal planet transit")
            trans_signal =  ellc.lc(t, t_zero=pars["t0"], period=pars["P"], radius_1=1/pars["aR"],
                                radius_2=pars["rp"]/pars["aR"], incl=pars["inc"],  sbratio=0, 
                                ld_1=self.LDC_model, ldc_1=pars["ld_pars"], shape_1='sphere', shape_2="love", 
                                grid_1=self.ellc_grid, grid_2=self.ellc_grid,q=pars["qmass"], hf_2=pars["hf"])
        else:
            # print("using batman for spherical planet transit")
            params = self._PCpars_to_batman(pars)
            m1 = batman.TransitModel(params, t)    #initializes model
            trans_signal = m1.light_curve(params)                    #calculates transit

        return trans_signal

    def eclipse_signal(self,t,pars):
        params = self._PCpars_to_batman(pars)
        m2 = batman.TransitModel(params, t, transittype="secondary")
        ecl_signal = self._rescale(m2.light_curve(params)) #ECLIPSE rescaled to 0-1
        return ecl_signal


    def _rescale(self,signal):
        #rescales signal to 0-1
        return (signal-min(signal))/np.ptp(signal) if np.all(min(signal) != max(signal)) else signal
    
    def _PCpars_to_batman(self,pars):
        params             = batman.TransitParams()       #object to store batman transit parameters
        params.t0          = pars["t0"]                #time of inferior conjunction
        params.per         = pars["P"]                #orbital period
        params.rp          = pars["rp"]                #planet radius (in units of stellar radii)
        params.a           = pars["aR"]                 #semi-major axis (in units of stellar radii)
        params.inc         = pars["inc"]              #orbital inclination (in degrees)
        params.ecc         = pars["e"]            #eccentricity
        params.w           = pars["w"]                #longitude of periastron (in degrees)
        params.limb_dark   = pars["LDC_model"]        #limb darkening model
        params.u           = pars["ld_pars"]          #limb darkening coefficients [u1, u2, u3, u4]
        params.fp          = pars["Fp"] *self.ppm               #planet flux
        params.t_secondary = params.t0 + 0.5*params.per   #time of secondary eclipse  #TODO: modify this for eccentric orbit
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

        tdur       = (pars["P"]/np.pi) * (ecc_fac**2/np.sqrt(1-pars["e"]**2)) * (np.arcsin( np.sqrt(1+pars["rp"]**2 - (pars["aR"]*ecc_fac*cosi)**2 )/denom ))
        return tdur

    def _phase_fold(self, t, pars):
        #phase folding
        phase = (t-pars["t0"])/pars["P"] % 1
        return phase   
    

    def phase_curve(self, t, pars, return_flux=False):
        self.phase = self._phase_fold(t,pars)
        phi      = 2*np.pi*self.phase
        self.phi = phi   

        trans_signal     = self.transit_signal(t, pars)
        ecl_signal       = self.eclipse_signal(t, pars)
        atm_signal       = self.planet_cosine_variation(t, pars)
        ellip_signal     = self.ellipsoidal_variation(t, pars)
        DB_signal        = self.doppler_beaming(t, pars)

        ellipsoidal_area = self.projected_ellipse_area(pars)
        normalized_area  = ellipsoidal_area/ellipsoidal_area.min()

        if return_flux:
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







