import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate
import scipy.interpolate as interpolate

#====================================================================
# Base Class
#====================================================================
class GammaMod(EosMod):
    """
    Abstract Equation of State class for Gruneisen Parameter curves
    """
    __metaclass__ = ABCMeta

    def __init__( self, V0ref=True ):
        self.V0ref = V0ref
        pass

    @abstractmethod
    def gamma( self, V_a, eos_d ):
        """Returns Gruneisen Param (gamma) variation due to compression."""

    # @abstractmethod
    # def temp( self, V_a, T0, eos_d ):
    #     """Returns Gruneisen Param (gamma) variation due to compression."""

    def temp( self, V_a, TR, eos_d ):
        """
        Return temperature for debye model
        V_a: sample volume array
        TR: temperature at V=VR
        """
        if np.isscalar(V_a):
            V_a = np.array([V_a])

        TOL = 1e-8
        Nsamp = 81
        # Nsamp = 281
        # Nsamp = 581

        if self.V0ref:
            VR, = Control.get_params( ['V0'], eos_d )
        else:
            VR, = Control.get_params( ['VR'], eos_d )


        Vmin = np.min(V_a)
        Vmax = np.max(V_a)

        dVmax = np.log(Vmax/VR)
        dVmin = np.log(Vmin/VR)

        T_a = TR*np.ones(V_a.size)

        if np.abs(dVmax) < TOL:
            dVmax = 0.0
        if np.abs(dVmin) < TOL:
            dVmin = 0.0


        if dVmax > TOL:
            indhi_a = np.where(np.log(V_a/VR) > TOL)[0]
            # indhi_a = np.where(V_a > VR)[0]

            # ensure numerical stability by shifting
            # if (Vmax-VR)<=TOL:
            #     T_a[indhi_a] = TR
            # else:
            Vhi_a = np.linspace(VR,Vmax,Nsamp)
            gammahi_a = self.gamma( Vhi_a, eos_d )
            logThi_a = integrate.cumtrapz(-gammahi_a/Vhi_a,x=Vhi_a)
            logThi_a = np.append([0],logThi_a)
            logtemphi_f = interpolate.interp1d(Vhi_a,logThi_a,kind='cubic')
            T_a[indhi_a] = TR*np.exp(logtemphi_f(V_a[indhi_a]))

        if dVmin < -TOL:
            indlo_a = np.where(np.log(V_a/VR) < -TOL)[0]
            # indlo_a = np.where(V_a <= VR)[0]

            # # ensure numerical stability by shifting
            # if (VR-Vmin)<TOL:
            #     T_a[indlo_a] = TR
            # else:
            Vlo_a = np.linspace(VR,Vmin,Nsamp)
            gammalo_a = self.gamma( Vlo_a, eos_d )
            logTlo_a = integrate.cumtrapz(-gammalo_a/Vlo_a,x=Vlo_a)
            logTlo_a = np.append([0],logTlo_a)
            logtemplo_f = interpolate.interp1d(Vlo_a,logTlo_a,kind='cubic')
            T_a[indlo_a] = TR*np.exp(logtemplo_f(V_a[indlo_a]))

        return T_a
#====================================================================

#====================================================================
# Implementations
#====================================================================
class GammaPowLaw(GammaMod):

    def __init__( self, V0ref=True, use_gammap=False ):
        self.V0ref = V0ref
        self.use_gammap = use_gammap
        pass

    def get_paramkey( self, eos_d ):
        if self.use_gammap:
            gammaderiv_typ = 'gammap'
        else:
            gammaderiv_typ = 'q'

        if self.V0ref:
            VRkey = 'V0'
            gammaRkey = 'gamma0'
            gammaderivkey = gammaderiv_typ+'0'
        else:
            VRkey = 'VR'
            gammaRkey = 'gammaR'
            gammaderivkey = gammaderiv_typ+'R'

        paramkey_a = [gammaRkey, gammaderivkey, VRkey]
        return paramkey_a

    def get_model_params( self, eos_d, ):
        paramkey_a = self.get_paramkey( eos_d )
        gammaR, gammaderiv, VR = Control.get_params(paramkey_a, eos_d)

        if self.use_gammap:
            qR = gammaderiv/gammaR
        else:
            qR = gammaderiv

        return ( gammaR, qR, VR )

    def get_param_scale_sub( self, eos_d ):
        """Return scale values for each parameter"""

        paramkey_a = self.get_paramkey( eos_d )
        gammaR, gammaderiv, VR = Control.get_params(paramkey_a, eos_d)

        gammaR_scl = 1.0
        VR_scl = VR
        # scale value for gammaderiv is equal to 1 for both gammap and q
        gammaderiv_scl = 1.0

        scale_a = np.array([gammaR_scl,gammaderiv_scl,VR_scl])

        return scale_a, paramkey_a

    def gamma( self, V_a, eos_d ):
        # OLD version fixed to zero-press ref volume
        # V0, gamma0, qR = Control.get_params( ['V0','gamma0','qR'], eos_d )
        # gamma_a = gamma0 *(V_a/V0)**qR

        # generalized version
        gammaR, qR, VR = self.get_model_params( eos_d )
        gamma_a = gammaR *(V_a/VR)**qR

        return gamma_a

    def temp( self, V_a, TR, eos_d ):
        """
        Return temperature for debye model
        V_a: sample volume array
        TR: temperature at V=VR
        """

        # generalized version
        gammaR, qR, VR = self.get_model_params( eos_d )
        gamma_a = self.gamma( V_a, eos_d )
        T_a = TR*np.exp( -(gamma_a - gammaR)/qR )

        return T_a
#====================================================================
class GammaFiniteStrain(GammaMod):
    def get_paramkey( self, eos_d ):
        if self.V0ref:
            VRkey = 'V0'
            gammaRkey = 'gamma0'
            gammapRkey = 'gammap0'
        else:
            VRkey = 'VR'
            gammaRkey = 'gammaR'
            gammapRkey = 'gammapR'

        paramkey_a = [gammaRkey, gammapRkey, VRkey]
        return paramkey_a

    def calc_strain_coefs( self, eos_d ):
        paramkey_a = self.get_paramkey( eos_d )
        gammaR, gammapR, VR = Control.get_params(paramkey_a, eos_d)

        a1 = 6*gammaR
        a2 = -12*gammaR +36*gammaR**2 -18*gammapR
        return a1, a2

    def get_param_scale_sub( self, eos_d ):
        """Return scale values for each parameter"""

        paramkey_a = self.get_paramkey( eos_d )
        gammaR, gammapR, VR = Control.get_params( paramkey_a, eos_d )

        gammaR_scl = 1.0
        gammapR_scl = 1.0
        VR_scl = VR

        scale_a = np.array([gammaR_scl,gammapR_scl,VR_scl])

        return scale_a, paramkey_a

    def calc_fstrain( self, V_a, eos_d ):
        paramkey_a = self.get_paramkey( eos_d )
        gammaR, gammapR, VR = Control.get_params(paramkey_a, eos_d)

        fstr = 0.5*((VR/V_a)**(2./3)-1.0)
        # print (V_a)
        # if np.any(np.isnan(V_a)):
        return fstr

    def gamma( self, V_a, eos_d ):
        a1, a2 = self.calc_strain_coefs( eos_d )
        fstr_a = self.calc_fstrain( V_a, eos_d )

        gamma_a = (2*fstr_a+1)*(a1+a2*fstr_a)/(6*(1+a1*fstr_a+0.5*a2*fstr_a**2))

        return gamma_a

    def temp( self, V_a, TR, eos_d ):
        a1, a2 = self.calc_strain_coefs( eos_d )
        fstr_a = self.calc_fstrain( V_a, eos_d )

        T_a = TR*np.sqrt(1 + a1*fstr_a + 0.5*a2*fstr_a**2)

        return T_a
#====================================================================

#
