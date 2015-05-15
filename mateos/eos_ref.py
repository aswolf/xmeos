from abc import ABCMeta, abstractmethod

#====================================================================
#      Reference Equation of States
#====================================================================
class EosRef(object):
    """
    Abstract Equation of State class for reference thermodynamic path
    """

    __metaclass__ = ABCMeta

    # #TODO:
    # * move fd_dcoor to object variable
    # * run set_fd_dcoor upon initialization
    # * try read dcoor val from param_d, otherwise use default values in
    # fd_dcoor

    coor_opt_d = {'E': 'energy', 'P': 'press', 'V': 'vol',
                  'T': 'temp', 'S': 'entropy'}

    @abstractmethod
    def __init__( self, prime_coor_id, const_coor_d, param_d=None ):
        self.param_d = param_d

        # validate and set prime and const coors
        check_coor_id( prime_coor_id )
        check_const_coor_d( const_coor_d )
        self.prime_coor_id = prime_coor_id
        self.const_coor_d = const_coor_d

        # set general func behavior based on prim and const coors
        set_prime_coor_funcs( prime_coor_id, const_coor_d )
        set_second_coor_funcs( prime_coor_id, const_coor_d )
        set_infer_coor_funcs( prime_coor_id, const_coor_d )
        set_fd_dcoor()

    def check_coor_id( coor_id ):
        coor_opt_id = self.coor_opt_d.keys()
        try:
            assert coor_id is str 'coor_id must be a string'
            assert coor_id in coor_opt_id 'coor_id must in coor_opt_id: '\
                + coor_opt_id
        except:
            [assert icoor_id is str 'coor_id must be a string'
             for icoor_id in coor_id]
            [assert icoor_id in coor_opt_id
             'coor_id must in coor_opt_d: ' + coor_opt_id
             for icoor_id in coor_id]
        pass

    def check_const_coor_d( const_coor_d ):
        try:
            [assert np.isscalar(val) or val is None for val in const_coor_id.values()]
        except Exception as e:
            print( 'const_coor_d is not valid' )
            raise e
        check_coor_id( const_coor_d.keys() )
        pass

    # Use globals()[str] = func() to set generic func behavior
    # Example of desired automatic behavior:
    # if prime_coor_id = 'V':
    #   * energy_V, press_V, temp_V are provided by subclasses
    #   * vol_E, vol_P, vol_T are provided numerically by inference
    #   * All others are provided by chained function calls
    #     (energy_P, energy_T, press_E, press_T, temp_E, temp_P)
    #      e.g., energy_P: return energy_V( vol_P( P_a ) )
    # for param( prime coor ):
    #   func provided by user model
    # for
    def set_prime_coor_funcs( prime_coor_id, const_coor_d ):
        pass

    def set_second_coor_funcs( prime_coor_id, const_coor_d ):
        pass

    def set_infer_coor_funcs( prime_coor_id, const_coor_d ):
        pass

    # def energy( coor_a, param_d=self.param_d ):
    #     globals()['energy_'+self.prime_coor_id]( coor_a, param_d )
    #     pass
    # def press( coor_a, param_d=self.param_d ):
    #     globals()['press_'+self.prime_coor_id]( coor_a, param_d )
    #     pass
    # def vol( coor_a, param_d=self.param_d ):
    #     globals()['vol_'+self.prime_coor_id]( coor_a, param_d )
    #     pass
    # def temp( coor_a, param_d=self.param_d ):
    #     globals()['temp_'+self.prime_coor_id]( coor_a, param_d )
    #     pass

    def energy_V( V_a, param_d=self.param_d ):
        return infer_coor( V_a, vol_E, param_d )
    def press_V( V_a, param_d=self.param_d ):
        return infer_coor( V_a, vol_P, param_d )
    def temp_V( V_a, param_d=self.param_d ):
        return infer_coor( V_a, vol_T, param_d )


    def energy_P( P_a, param_d=self.param_d ):
        return infer_coor( P_a, press_E, param_d )
    def vol_P( P_a, param_d=self.param_d ):
        return infer_coor( P_a, press_V, param_d )
    def temp_P( P_a, param_d=self.param_d ):
        return infer_coor( P_a, press_T, param_d )

    def energy_T( T_a, param_d=self.param_d ):
        return infer_coor( T_a, temp_E, param_d )
    def press_T( T_a, param_d=self.param_d ):
        return infer_coor( T_a, temp_P, param_d )
    def vol_T( T_a, param_d=self.param_d ):
        return infer_coor( T_a, temp_V, param_d )

    def press_E( E_a, param_d=self.param_d ):
        return infer_coor( E_a, energy_P, param_d )
    def vol_E( E_a, param_d=self.param_d ):
        return infer_coor( E_a, energy_V, param_d )
    def temp_E( E_a, param_d=self.param_d ):
        return infer_coor( E_a, energy_T, param_d )


    def infer_coor( coor0_a, coor0_func, param_d ):
        try:
            # coor0_a - coor0_func(coor_a) = 0
        except RuntimeError as err:
            print("Required EOS functions not defined for this operation. \
                  One of the pair of conjugate functions must be defined.\
                  For instance either press_V() or vol_P() must be defined \
                  to enable inference.")
            raise err

    def set_fd_dcoor( tol=1e-4 )
        Tscl_bnds = [300, 3000]
        Pscl_bnds = [0, 200]

        dT = tol*np.abs(np.diff(Tscl))
        dP = tol*np.abs(np.diff(Pscl))

        try:
            dE_P = tol*np.abs( np.diff( energy_P( np.array([ P0, Pscl ]),
                                                 param_d )))
        except:
            dE_P = 0

        try:
            dE_T = tol*np.abs( np.diff( energy_T( np.array([ T0, Tscl ]),
                                                 param_d )))
        except:
            dE_T = 0

        try:
            dE_V = np.abs( np.diff( energy_V( np.array([ V0, (1-tol)*V0 ]),
                                             param_d )))
        except:
            dE_V = 0

        dE = np.amax( [dE_P, dE_T, dE_V] )

        self.fd_dcoor = {'dE': dE, 'dP': dP, 'dV': dV, 'dT': dT}

    def deriv_fd_E( E_a, coor_func_E, param_d ):
        dE = param_d['fd_dcoor']['dE']
        return deriv_fd( E_a, coor_func_E, dE, param_d )

    def deriv_fd_P( P_a, coor_func_P, param_d ):
        dP = param_d['fd_dcoor']['dP']
        return deriv_fd( P_a, coor_func_P, dP, param_d )

    def deriv_fd_V( V_a, coor_func_V, param_d ):
        dV = param_d['fd_dcoor']['dV']
        return deriv_fd( V_a, coor_func_V, dV, param_d )

    def deriv_fd_T( T_a, coor_func_T, param_d ):
        dT = param_d['fd_dcoor']['dT']
        return deriv_fd( T_a, coor_func_T, dT, param_d )

    def deriv_fd( coor_a, coor_func, dcoor, param_d ):
        """
        Use 5-pt stencil method to return finite diff deriv
        """
        fx_2hi = coor_func( coor_a + 2*dcoor, param_d )
        fx_1hi = coor_func( coor_a + 1*dcoor, param_d )
        fx_1lo = coor_func( coor_a - 1*dcoor, param_d )
        fx_2lo = coor_func( coor_a - 2*dcoor, param_d )
        dfdx = (-fx_2hi + 8*fx_1hi -8*fx_1lo + fx_2lo)/(12*dcoor)
        return dfdx

#    eosref.EosIsotherm( eosref.Vinet, param_d=param_d, T0=None )
#    eosref.EosAdiabat( eosref.BirchMurn3, param_d=param_d, T0=None )
#    eosref.EosIsobar( eosref.ThermExpPoly, param_d=param_d, P0=None )
#    eosref.EosIsochore( eosref.RosenTarazona, param_d=param_d, V0=None )
#    eosref.EosHugoniot( eosref.LinearShock, param_d=param_d, V0=None, T0=None )
#    eosref.EosLiquidus( eosref.ClausiusClapeyron, param_d=param_d, Tliq0=None )
#    eosref.EosSolidus( eosref.ClausiusClapeyron, param_d=param_d, Tsol0=None )
#
#    eosref.EosModel( eosref.Vinet, param_d=param_d )

#====================================================================
# EosCompress- Isotherm: E(V), P(V), T=const
# EosCompress- Adiabat: E(V), P(V), [T(V)], S=const_unknown
# EosHeat- Isobar: V(T), [E(T)], P=const
# EosHeat- Isochore: E(T), [P(T)], V=const
# EosCompress- Hugoniot: E(V), P(V)
# EosCompress- Liquidus: {P(V), T(V), [S(V)]} OR {V(P), T(P), [S(P)]}
# EosCompress- Solidus: {P(V), T(V), [S(V)]} OR {V(P), T(P), [S(P)]}
#====================================================================
class EosIsotherm( EosRef ):
    def __init__( self, eos_mod, T0=300, param_d=None ):
        assert hasattr(eos_mod, 'energy_V') \
            'eos_mod must implement energy_V( V_a, param_d )'
        assert hasattr(eos_mod, 'press_V') \
            'eos_mod must implement press_V( V_a, param_d )'
        self.eos_mod = eos_mod
        self.param_d = param_d
        self.T0 = T0

    def energy( V_a, param_d=self.param_d ):
        return energy_V( V_a, param_d )
    def press( V_a, param_d=self.param_d ):
        return press_V( V_a, param_d )
    def temp( V_a, param_d=self.param_d ):
        return temp_V( V_a, param_d )
    def vol( V_a, param_d=self.param_d ):
        return V_a
    def bulk_mod( V_a, param_d=self.param_d ):
        return bulk_mod_T( V_a, param_d )

    def energy_V( V_a, param_d=self.param_d ):
        return self.eos_mod.energy_V( V_a, param_d )

    def press_V( V_a, param_d=self.param_d ):
        return self.eos_mod.press_V( V_a, param_d )

    def temp_V( V_a, param_d=self.param_d ):
        return self.T0*np.ones( V_a.shape )

    def energy_T( T_a, param_d=self.param_d ):
        raise NotImplementedError()
    def press_T( T_a, param_d=self.param_d ):
        raise NotImplementedError()
    def vol_T( T_a, param_d=self.param_d ):
        raise NotImplementedError()

    def bulk_mod_T( V_a, param_d=self.param_d ):
        try:
            self.eos_mod.bulk_mod( V_a, param_d )
        except:
            self.bulk_mod_T( V_a, param_d )




class EosAdiabat( EosPath ):

class EosIsobar( EosPath ):

class EosIsochore( EosPath ):

class EosHugoniot( EosPath ):

class EosLiquidus( EosPath ):

class EosSolidus( EosPath ):

class EosCompress( EosRef ):
    """
    Abstract class to describe a reference Equation of State of a
    compressive thermodynamic pathway
    """

    __metaclass__ = ABCMeta

    def __init__( self, param_d=None ):
        self.param_d = param_d
        pass

    @abstractmethod
    def energy_V( V_a, param_d=param_d ):
        pass
    @abstractmethod
    def press_V( V_a, param_d=param_d ):
        pass


    def energy( V_a, param_d=self.param_d ):
        return energy_V( V_a, param_d=param_d )

    def press( V_a, param_d=self.param_d ):
        return press_V( V_a, param_d=param_d )

    def vol( P_a, param_d=self.param_d ):
        return vol_P( P_a, param_d=param_d )

    def temp( V_a, param_d=self.param_d ):
        return None

#====================================================================
class EosHeat( EosRef ):
    """
    Abstract class to describe a reference Equation of State
    of a heating thermodynamic pathway
    """

    __metaclass__ = ABCMeta

    def __init__( self, param_d=None ):
        self.param_d = param_d
        pass

    @abstractmethod
    def vol_T( T_a, param_d=param_d ):
        pass
    @abstractmethod
    def energy_T( T_a, param_d=param_d ):
        pass

    def vol( T_a, param_d=self.param_d ):
        return vol_T( T_a, param_d=param_d )

    def energy( T_a, param_d=self.param_d ):
        return energy_T( T_a, param_d=param_d )

    def temp( V_a, param_d=self.param_d ):
        return temp_V( V_a, param_d=param_d )

    def press( T_a, param_d=self.param_d ):
        return None



#====================================================================
class EosModel( EosRef ):
    """
    Abstract class defining specific thermodynamic pathway
    """

    __metaclass__ = ABCMeta

    # What are needed extra methods?

    @abstractmethod
    def set_eos_mod( eos ):
        self.eos_mod()

    # EosCompress, EosHeat
    # EosPath



#====================================================================
#     EosCompress Models
#====================================================================
class BirchMurn3( EosCompress ):
    """
    Abstract Isothermal Equation of State class
    """

    def __init__( self, param_d=None ):
        self.param_d = param_d
        pass

    def energy_V( V_a, param_d=self.param_d ):
        V0 = param_d['V0']
        K0 = param_d['K0']
        K0p = param_d['K0p']
        PV_const = param_d['PV_const']

        vratio_a = V_a/V0

        fstrain_a = 0.5*(vratio_a**(-2.0/3) - 1)

        energy_a = 9.0/2*V0*K0*PV_const*\
            ( K0p*fstrain_a**3 + fstrain_a**2*(1-4*fstrain_a) )

        return energy_a

    def press_V( V_a, param_d=self.param_d ):
        V0 = param_d['V0']
        K0 = param_d['K0']
        K0p = param_d['K0p']

        vratio_a = V_a/V0

        press_a = 3.0/2*K0 * (vratio_a**(-7.0/3) - vratio_a**(-5.0/3)) * \
            (1 + 3.0/4*(K0p-4)*(vratio_a**(-2.0/3)-1))

        return press_a

#====================================================================
class Vinet( EosCompress ):
    """
    Abstract Isothermal Equation of State class
    """

    def __init__( self, param_d=None ):
        self.param_d = param_d
        pass

    def energy_V( V_a, param_d=self.param_d ):
        raise NotImplementedError( "Need to implement!" )
        pass

    def press_V( V_a, param_d=self.param_d ):
        raise NotImplementedError( "Need to implement!" )
        pass


#====================================================================
# EosCompress- Isotherm: E(V), P(V), T=const
# EosCompress- Adiabat: E(V), P(V), [T(V)], S=const_unknown
# EosHeat- Isobar: V(T), [E(T)], P=const
# EosHeat- Isochore: E(T), [P(T)], V=const
# EosCompress- Hugoniot: E(V), P(V)
# EosCompress- Liquidus: {P(V), T(V), [S(V)]} OR {V(P), T(P), [S(P)]}
# EosCompress- Solidus: {P(V), T(V), [S(V)]} OR {V(P), T(P), [S(P)]}
#====================================================================

class EosIsotherm( EosPath ):
    def __init__( self, eos_compress, param_d=None ):
        self.param_d = param_d
        self.eos_mod = eos_compress
        pass

class EosAdiabat( EosPath ):

class EosIsobar( EosPath ):

class EosIsochore( EosPath ):

class EosHugoniot( EosPath ):

class EosLiquidus( EosPath ):

class EosSolidus( EosPath ):


#     @abstractmethod
#     def bulk_mod( V_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def k_prime( V_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def energy( VT_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def press( VT_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def entropy( VT_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def gamma( VT_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def therm_exp( VT_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def heat_capacity_V( VT_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def heat_capacity_P( VT_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def bulk_mod_T( VT_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def bulk_mod_S( VT_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def k_prime_T( VT_a, param_d=self.param_d ):
#         pass
#
#     @abstractmethod
#     def k_prime_S( VT_a, param_d=self.param_d ):
#         pass
#
