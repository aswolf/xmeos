import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from scipy import integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

#====================================================================
#     xmeos: Xtal-Melt Equation of State package
#      models - library of equation of state models
#====================================================================

#====================================================================
# SECT 0: Admin functions
#====================================================================
class Control(object):
    @classmethod
    def init_consts( cls, eos_d ):
        eos_d['const_d'] = cls.default_consts()
        pass

    @classmethod
    def default_consts(cls):
        const_d = {}
        const_d['eVperHa'] = 27.211 # eV/Ha
        const_d['JperHa'] = 4.35974434e-18 # J/Ha
        const_d['JperCal'] = 4.184 # J/Cal
        const_d['Nmol'] = 6.0221413e+23 # atoms/mol
        const_d['kJ_molpereV'] = 96.49 # kJ/mol/eV
        const_d['R'] = 8.314462 # J/K/mol
        const_d['kboltz'] = 8.617332e-5 # eV/K
        const_d['ang3percc'] = 1e24 # ang^3/cm^3

        const_d['PV_ratio'] = 160.2176487 # (GPa*ang^3)/eV
        const_d['TS_ratio'] = const_d['R']/const_d['kboltz'] # (J/mol)/eV

        return const_d

    @classmethod
    def set_consts( cls, name_l, val_l, eos_d ):
        if 'const_d' in eos_d.keys():
            const_d = eos_d['const_d']
        else:
            cls.init_consts( eos_d )

        for name, val in zip( name_l, val_l ):
            const_d[name] = val

        pass

    @classmethod
    def get_consts( cls, name_l, eos_d ):
        """
        Retrieve list of desired consts stored in eos_d['const_d']
        """
        const_d = eos_d['const_d']
        const_l = []
        for name in name_l:
            const_l.append( const_d[name] )

        return tuple( const_l )

    @classmethod
    def set_params( cls, name_l, val_l, eos_d ):
        if 'param_d' in eos_d.keys():
            param_d = eos_d['param_d']
        else:
            param_d = {}
            eos_d['param_d'] = param_d

        for name, val in zip( name_l, val_l ):
            param_d[name] = val

        pass

    @classmethod
    def get_params( cls, name_l, eos_d ):
        """
        Retrieve list of desired params stored in eos_d['param_d']
        """
        param_d = eos_d['param_d']
        param_l = []
        for name in name_l:
            param_l.append( param_d[name] )

        return tuple( param_l )

    @classmethod
    def swap_params( cls, name_l, eos_d ):
        """
        Retrieve list of desired params stored in eos_d['param_d']
        """

        # Use shallow copy to avoid unneeded duplication
        eos_swap_d = copy.copy( eos_d )
        # Use deep copy on params to ensure swap without affecting original
        param_swap_d = copy.deepcopy(eos_d['param_d'])

        eos_swap_d['param_d'] = param_swap_d

        cls.set_params( name_l, eos_swap_d )

        return eos_swap_d

    @classmethod
    def set_array_params( cls, basename, param_arr_a, eos_d ):
        name_l = []

        for i in range(len(param_arr_a)):
            iname = basename+'_'+np.str(i)
            name_l.append(iname)

        cls.set_params(name_l, param_arr_a, eos_d)

    @classmethod
    def get_array_params( cls, basename, eos_d ):
        param_d = eos_d['param_d']
        paramkeys_a = np.array(param_d.keys())

        baselen = len(basename+'_')

        mask = np.array([key.startswith(basename+'_') for key in paramkeys_a])

        arrindlist = []
        vallist = []
        for key in paramkeys_a[mask]:
            idstr = key[baselen:]
            try:
                idnum = np.array(idstr).astype(np.float)
                assert np.equal(np.mod(idnum,1),0), \
                    'Parameter keys that are part of a parameter array must '+\
                    'have form "basename_???" where ??? are integers.'
                idnum = idnum.astype(np.int)
            except:
                assert False, 'That basename does not correspond to any valid parameter arrays stored in eos_d'

            arrindlist.append(idnum)
            vallist.append(param_d[key])

        arrind_a = np.array(arrindlist)
        val_a = np.array(vallist)

        if arrind_a.size==0:
            return np.array([])
        else:
            indmax = np.max(arrind_a)

            param_arr = np.zeros(indmax+1)
            for arrind, val in zip(arrind_a,val_a):
                param_arr[arrind] = val

            return param_arr

    @classmethod
    def set_modtypes( cls, name_l, val_l, eos_d ):
        if 'modtype_d' in eos_d.keys():
            modtype_d = eos_d['modtype_d']
        else:
            modtype_d = {}
            eos_d['modtype_d'] = modtype_d

        # Should we verify match?
        for name, val in zip( name_l, val_l ):
            if globals().has_key(name):
                # modtype = globals()[name]
                # modtype_d[name] = modtype()
                modtype_d[name] = val
            else:
                print name + " is not a valid modtype object"

        pass

    @classmethod
    def get_modtypes( cls, name_l, eos_d ):
        """
        Retrieve list of desired model types stored in eos_d['modtype_d']
        """
        modtype_d = eos_d['modtype_d']
        modtype_l = []
        for name in name_l:
            modtype_l.append( modtype_d[name] )

        return tuple( modtype_l )

    @classmethod
    def set_args( cls, name_l, val_l, eos_d ):
        if 'arg_d' in eos_d.keys():
            arg_d = eos_d['arg_d']
        else:
            arg_d = {}
            eos_d['arg_d'] = arg_d

        for name, val in zip( name_l, val_l ):
            arg_d[name] = val

        pass
#====================================================================

#====================================================================
# SECT 1: Fitting Routines
#====================================================================
class ModFit(object):
    def __init__( self ):
        pass

    def get_resid_fun( self, eos_fun, eos_d, param_key_a, sys_state_tup,
                      val_a, err_a=1.0):

        # def resid_fun( param_in_a, eos_fun=eos_fun, eos_d=eos_d,
        #               param_key_a=param_key_a, sys_state_tup=sys_state_tup,
        #               val_a=val_a, err_a=err_a ):

        #     # Don't worry about transformations right now
        #     param_a = param_in_a

        #     # set param value in eos_d dict
        #     globals()['set_param']( param_key_a, param_a, eos_d )

        #     # Take advantage of eos model function input format
        #     #   uses tuple expansion for input arguments
        #     mod_val_a = eos_fun( *(sys_state_tup+(eos_d,)) )
        #     resid_a = (mod_val_a - val_a)/err_a
        #     return resid_a

        wrap_eos_fun = self.get_wrap_eos_fun( eos_fun, eos_d, param_key_a )

        def resid_fun( param_in_a, wrap_eos_fun=wrap_eos_fun,
                      sys_state_tup=sys_state_tup,
                      val_a=val_a, err_a=err_a ):

            mod_val_a = wrap_eos_fun( param_in_a, sys_state_tup )
            resid_a = (mod_val_a - val_a)/err_a
            return resid_a

        return resid_fun

    def get_wrap_eos_fun( self, eos_fun, eos_d, param_key_a ):

        def wrap_eos_fun(param_in_a, sys_state_tup, eos_fun=eos_fun,
                         eos_d=eos_d, param_key_a=param_key_a ):

            # Don't worry about transformations right now
            param_a = param_in_a

            # set param value in eos_d dict
            Control.set_params( param_key_a, param_a, eos_d )

            # Take advantage of eos model function input format
            #   uses tuple expansion for input arguments
            mod_val_a = eos_fun( *(sys_state_tup+(eos_d,)) )
            return mod_val_a

        return wrap_eos_fun
#====================================================================
# SECT N: Code Utility Functions
#====================================================================
def fill_array( var1, var2 ):
    """
    fix fill_array such that it returns two numpy arrays of equal size

    use numpy.full_like

    """
    var1_a = np.asarray( var1 )
    var2_a = np.asarray( var2 )

    if var1_a.shape==():
        var1_a = np.asarray( [var1] )
    if var2_a.shape==():
        var2_a = np.asarray( [var2] )

    # Begin try/except block to handle all cases for filling an array
    while True:
        try:
            assert var1_a.shape == var2_a.shape
            break
        except: pass
        try:
            var1_a = np.full_like( var2_a, var1_a )
            break
        except: pass
        try:
            var2_a = np.full_like( var1_a, var2_a )
            break
        except: pass

        # If none of the cases properly handle it, throw error
        assert False, 'var1 and var2 must both be equal shape or size=1'

    return var1_a, var2_a
#====================================================================

#====================================================================
# SECT 3: EOS  Objects
#====================================================================
# 3.1: Base Clases
#====================================================================
class EosMod(object):
    """
    Abstract Equation of State Parent Base class
    """
    __metaclass__ = ABCMeta

    def __init__( self ):
        pass

    def get_param_scale( self, eos_d, apply_expand_adj=True):
        """Return scale values for each parameter"""
        return self.get_param_scale_sub( eos_d )

    def get_param_scale_sub( self, eos_d ):
        raise NotImplementedError("'get_param_scale_sub' function not implimented for this model")
#====================================================================
class CompressMod(EosMod):
    """
    Abstract Equation of State class to describe Compression Behavior

    generally depends on both vol and temp
    """
    __metaclass__ = ABCMeta

    def press( self, V_a, T_a, eos_d ):
        """Returns Press behavior due to compression."""
        return self.calc_press(V_a, T_a, eos_d)

    def vol( self, P_a, T_a, eos_d ):
        """Returns Vol behavior due to compression."""
        return self.calc_vol(V_a, T_a, eos_d)

    def energy( self, V_a, T_a, eos_d ):
        """Returns Energy behavior due to compression."""
        return self.calc_energy(V_a, T_a, eos_d)

    def energy_perturb( self, V_a, T_a, eos_d ):
        """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""
        return self.calc_energy_perturb(V_a, T_a, eos_d)

    def bulk_mod( self, V_a, T_a, eos_d ):
        """Returns Bulk Modulus behavior due to compression."""
        return self.calc_bulk_mod(V_a, T_a, eos_d)

    def bulk_mod_deriv( self, V_a, T_a, eos_d ):
        """Returns Bulk Modulus Deriv (K') behavior due to compression."""
        return self.calc_bulk_mod_deriv(V_a, T_a, eos_d)

    # Standard methods must be overridden (as needed) by implimentation model
    def calc_press( self, V_a, T_a, eos_d ):
        """Calculates Press behavior due to compression."""
        raise NotImplementedError("'calc_press' function not implimented for this model")

    def calc_vol( self, P_a, T_a, eos_d ):
        """Calculates Vol behavior due to compression."""
        raise NotImplementedError("'calc_vol' function not implimented for this model")

    def calc_energy( self, V_a, T_a, eos_d ):
        """Calculates Energy behavior due to compression."""
        raise NotImplementedError("'calc_energy' function not implimented for this model")

    def calc_energy_perturb( self, V_a, T_a, eos_d ):
        """Calculates Energy pertubation basis functions resulting from fractional changes to EOS params."""
        raise NotImplementedError("'calc_energy_perturb' function not implimented for this model")

    def calc_bulk_mod( self, V_a, T_a, eos_d ):
        """Calculates Bulk Modulus behavior due to compression."""
        raise NotImplementedError("'calc_bulk_mod' function not implimented for this model")

    def calc_bulk_mod_deriv( self, V_a, T_a, eos_d ):
        """Calculates Bulk Modulus Deriv (K') behavior due to compression."""
        raise NotImplementedError("'calc_bulk_mod_deriv' function not implimented for this model")
#====================================================================
class CompressPathMod(CompressMod):
    """
    Abstract Equation of State class for a reference Compression Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume
    """
    __metaclass__ = ABCMeta

    path_opts = ['T','S']
    supress_energy = False
    supress_press = False

    def __init__( self, path_const='T', level_const=300, expand_adj_mod=None,
                 expand_adj=None, supress_energy=False, supress_press=False ):
        assert path_const in self.path_opts, path_const + ' is not a valid ' + \
            'path const. You must select one of: ' + path_opts
        self.path_const = path_const
        self.level_const = level_const
        self.supress_energy = supress_energy
        self.supress_press = supress_press

        # Use Expansion Adjustment for negative pressure region?
        if expand_adj is None:
            self.expand_adj = False
        else:
            self.expand_adj = expand_adj

        if expand_adj_mod is None:
            self.expand_adj = False
            self.expand_adj_mod = None
        else:
            self.expand_adj = True
            self.expand_adj_mod = expand_adj_mod
        pass

    def validate_shared_param_scale( self, scale_pos_a, paramkey_pos_a,
                                    scale_neg_a, paramkey_neg_a ):
        TOL = 1e-4
        assert np.all(np.in1d(paramkey_pos_a,paramkey_neg_a)),\
            'paramkey_neg_a must be a superset of paramkey_pos_a'
        assert len(paramkey_neg_a) <= len(paramkey_pos_a)+1,\
            'paramkey_neg_a must have at most one more parameter than paramkey_neg_a'

        # shared_mask = np.in1d(paramkey_neg_a,paramkey_pos_a)
        # paramkey_shared_a = paramkey_neg_a[shared_mask]
        # scale_shared_a = scale_neg_a[shared_mask]

        ind_pos_a = np.array([np.where(paramkey_neg_a==paramkey)[0][0] \
                              for paramkey in paramkey_pos_a])
        # scale_a[ind_pos_a] = scale_pos_a

        assert np.all(np.log(scale_neg_a[ind_pos_a]/scale_pos_a)<TOL),\
            'Shared param scales must match to within TOL.'

        return ind_pos_a

    def get_param_scale( self, eos_d, apply_expand_adj=False , output_ind=False):
        if not self.expand_adj :
            return self.get_param_scale_sub( eos_d )
        else:
            scale_pos_a, paramkey_pos_a = self.get_param_scale_sub( eos_d )
            scale_neg_a, paramkey_neg_a = self.expand_adj_mod.get_param_scale_sub( eos_d )

            ind_pos_a = self.validate_shared_param_scale(scale_pos_a,paramkey_pos_a,
                                                         scale_neg_a,paramkey_neg_a)

            # Since negative expansion EOS model params are a superset of those
            # required for the positive compression model, we can simply return the
            # scale and paramkey values from the negative expansion model
            scale_a = scale_neg_a
            paramkey_a = paramkey_neg_a

            if output_ind:
                return scale_a, paramkey_a, ind_pos_a
            else:
                return scale_a, paramkey_a

    def get_ind_exp( self, V_a, eos_d ):
        V0 = Control.get_params( ['V0'], eos_d )
        ind_exp = np.where( V_a > V0 )[0]
        return ind_exp

    def get_path_const( self ):
        return self.path_const

    def get_level_const( self ):
        return self.level_const

    # EOS property functions
    def param_deriv( self, fname, paramname, V_a, eos_d, dxfrac=0.01):
        scale_a, paramkey_a = self.get_param_scale( eos_d, apply_expand_adj=True )
        scale = scale_a[paramkey_a==paramname][0]
        # print 'scale: ' + np.str(scale)

        #if (paramname is 'E0') and (fname is 'energy'):
        #    return np.ones(V_a.shape)
        try:
            fun = getattr(self, fname)
            # Note that self is implicitly included
            val0_a = fun( V_a, eos_d)
        except:
            assert False, 'That is not a valid function name ' + \
                '(e.g. it should be press or energy)'

        try:
            param = Control.get_params( [paramname], eos_d )[0]
            dparam = scale*dxfrac
            # print 'param: ' + np.str(param)
            # print 'dparam: ' + np.str(dparam)
        except:
            assert False, 'This is not a valid parameter name'

        # set param value in eos_d dict
        Control.set_params( [paramname,], [param+dparam,], eos_d )

        # Note that self is implicitly included
        dval_a = fun(V_a, eos_d) - val0_a

        # reset param to original value
        Control.set_params( [paramname], [param], eos_d )

        deriv_a = dval_a/dxfrac
        return deriv_a

    def press( self, V_a, eos_d, apply_expand_adj=True):
        if self.supress_press:
            zero_a = 0.0*V_a
            return zero_a

        else:
            press_a = self.calc_press(V_a, eos_d)
            if self.expand_adj and apply_expand_adj:
                ind_exp = self.get_ind_exp(V_a, eos_d)
                if (ind_exp.size>0):
                    press_a[ind_exp] = self.expand_adj_mod.calc_press( V_a[ind_exp], eos_d )

            return press_a
        pass

    def energy( self, V_a, eos_d, apply_expand_adj=True ):
        if self.supress_energy:
            zero_a = 0.0*V_a
            return zero_a

        else:
            energy_a =  self.calc_energy(V_a, eos_d)
            if self.expand_adj and apply_expand_adj:
                ind_exp = self.get_ind_exp(V_a, eos_d)
                if apply_expand_adj and (ind_exp.size>0):
                    energy_a[ind_exp] = self.expand_adj_mod.calc_energy( V_a[ind_exp], eos_d )

            return energy_a

    def bulk_mod( self, V_a, eos_d, apply_expand_adj=True ):
        bulk_mod_a =  self.calc_bulk_mod(V_a, eos_d)
        if self.expand_adj and apply_expand_adj:
            ind_exp = self.get_ind_exp(V_a, eos_d)
            if apply_expand_adj and (ind_exp.size>0):
                bulk_mod_a[ind_exp] = self.expand_adj_mod.calc_bulk_mod( V_a[ind_exp], eos_d )

        return bulk_mod_a

    def bulk_mod_deriv(  self,V_a, eos_d, apply_expand_adj=True ):
        bulk_mod_deriv_a =  self.calc_bulk_mod_deriv(V_a, eos_d)
        if self.expand_adj and apply_expand_adj:
            ind_exp = self.get_ind_exp(V_a, eos_d)
            if apply_expand_adj and (ind_exp.size>0):
                bulk_mod_deriv_a[ind_exp] = self.expand_adj_mod_deriv.calc_bulk_mod_deriv( V_a[ind_exp], eos_d )

        return bulk_mod_deriv_a

    def energy_perturb( self, V_a, eos_d, apply_expand_adj=True ):
        # Eval positive press values
        Eperturb_pos_a, scale_a, paramkey_a  = self.calc_energy_perturb( V_a, eos_d )

        if (self.expand_adj==False) or (apply_expand_adj==False):
            return Eperturb_pos_a, scale_a, paramkey_a
        else:
            Nparam_pos = Eperturb_pos_a.shape[0]

            scale_a, paramkey_a, ind_pos = \
                self.get_param_scale( eos_d, apply_expand_adj=True,
                                     output_ind=True )

            Eperturb_a = np.zeros((paramkey_a.size, V_a.size))
            Eperturb_a[ind_pos,:] = Eperturb_pos_a

            # Overwrite negative pressure Expansion regions
            ind_exp = self.get_ind_exp(V_a, eos_d)
            if ind_exp.size>0:
                Eperturb_adj_a = \
                    self.expand_adj_mod.calc_energy_perturb( V_a[ind_exp],
                                                            eos_d )[0]
                Eperturb_a[:,ind_exp] = Eperturb_adj_a

            return Eperturb_a, scale_a, paramkey_a

    #   Standard methods must be overridden (as needed) by implimentation model

    def get_param_scale_sub( self, eos_d):
        raise NotImplementedError("'get_param_scale_sub' function not implimented for this model")

    def calc_press( self, V_a, eos_d ):
        """Returns Press variation along compression curve."""
        raise NotImplementedError("'press' function not implimented for this model")

    def calc_energy( self, V_a, eos_d ):
        """Returns Energy along compression curve."""
        raise NotImplementedError("'energy' function not implimented for this model")

    def calc_energy_perturb( self, V_a, eos_d ):
        """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""

        fname = 'energy'
        scale_a, paramkey_a = self.get_param_scale\
            ( eos_d, apply_expand_adj=self.expand_adj )
        Eperturb_a = []
        for paramname in paramkey_a:
            iEperturb_a = self.param_deriv( fname, paramname, V_a, eos_d)
            Eperturb_a.append(iEperturb_a)

        Eperturb_a = np.array(Eperturb_a)

        return Eperturb_a, scale_a, paramkey_a

    def calc_bulk_mod( self, V_a, eos_d ):
        """Returns Bulk Modulus variation along compression curve."""
        raise NotImplementedError("'bulk_mod' function not implimented for this model")

    def calc_bulk_mod_deriv( self, V_a, eos_d ):
        """Returns Bulk Modulus Deriv (K') variation along compression curve."""
        raise NotImplementedError("'bulk_mod_deriv' function not implimented for this model")
#====================================================================
class ThermalMod(EosMod):
    """
    Abstract Equation of State class to describe Thermal Behavior

    generally depends on both vol and temp
    """

    __metaclass__ = ABCMeta

    # EOS property functions
    def energy( self, V_a, T_a, eos_d ):
        return self.calc_energy( V_a, T_a, eos_d )

    def heat_capacity( self, V_a, T_a, eos_d ):
        return self.calc_heat_capacity( V_a, T_a, eos_d )

    def press( self, V_a, T_a, eos_d ):
        return self.calc_press( V_a, T_a, eos_d )

    def entropy( self, V_a, T_a, eos_d ):
        return self.calc_entropy( V_a, T_a, eos_d )

    def vol( self, P_a, T_a, eos_d ):
        return self.calc_vol( P_a, T_a, eos_d )

    # Standard methods must be overridden (as needed) by implimentation model
    def calc_energy( self, V_a, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
        raise NotImplementedError("'energy' function not implimented for this model")

    def calc_heat_capacity( self, V_a, T_a, eos_d ):
        """Returns Heat Capacity."""
        raise NotImplementedError("'heat_capacity' function not implimented for this model")

    def calc_entropy( self, V_a, T_a, eos_d ):
        """Returns Entropy."""
        raise NotImplementedError("'entropy' function not implimented for this model")

    def calc_press( self, V_a, T_a, eos_d ):
        """Returns thermal contribution to pressure."""
        raise NotImplementedError("'press' function not implimented for this model")

    def calc_vol( self, V_a, T_a, eos_d ):
        """Returns thermally expanded volume."""
        raise NotImplementedError("'vol' function not implimented for this model")
#====================================================================
class ThermalPathMod(ThermalMod):
    """
    Abstract Equation of State class for a reference Thermal Path

    Path can either be isobaric (P=const) or isochoric (V=const)

    For this restricted path, thermodyn properties depend only on temperature.
    """

    __metaclass__ = ABCMeta

    path_opts = ['P','V']

    def __init__( self, path_const='V', level_const=np.nan ):
        assert path_const in self.path_opts, path_const + ' is not a valid ' + \
            'path const. You must select one of: ' + path_opts
        self.path_const = path_const
        self.level_const = level_const
        pass

    def get_path_const( self ):
        return self.path_const

    def get_level_const( self ):
        return self.level_const

    # EOS property functions
    def energy( self, T_a, eos_d ):
        return self.calc_energy( T_a, eos_d )

    def heat_capacity( self, T_a, eos_d ):
        return self.calc_heat_capacity( T_a, eos_d )

    def press( self, T_a, eos_d ):
        return self.calc_press( T_a, eos_d )

    def entropy( self, T_a, eos_d ):
        return self.calc_entropy( T_a, eos_d )

    def vol( self, T_a, eos_d ):
        return self.calc_vol( T_a, eos_d )

    # Standard methods must be overridden (as needed) by implimentation model
    def calc_energy( self, T_a, eos_d ):
        """Returns Thermal Component of Energy along heating path."""
        raise NotImplementedError("'energy' function not implimented for this model")

    def calc_heat_capacity( self, T_a, eos_d ):
        """Returns Heat Capacity along heating path."""
        raise NotImplementedError("'heat_capacity' function not implimented for this model")

    def calc_entropy( self, T_a, eos_d ):
        """Returns Entropy along heating path."""
        raise NotImplementedError("'entropy' function not implimented for this model")

    def calc_press( self, T_a, eos_d ):
        """Returns thermal contribution to pressure along heating path."""
        raise NotImplementedError("'press' function not implimented for this model")

    def calc_vol( self, T_a, eos_d ):
        """Returns thermally expanded volume along heating path."""
        raise NotImplementedError("'vol' function not implimented for this model")
#====================================================================
class GammaMod(EosMod):
    """
    Abstract Equation of State class for Gruneisen Parameter curves
    """
    __metaclass__ = ABCMeta

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
        Nsamp = 81

        if self.V0ref:
            VR, = Control.get_params( ['V0'], eos_d )
        else:
            VR, = Control.get_params( ['VR'], eos_d )


        Vmin = np.min(V_a)
        Vmax = np.max(V_a)

        T_a = np.zeros(V_a.size)

        if Vmax > VR:
            Vhi_a = np.linspace(VR,Vmax,Nsamp)
            gammahi_a = self.gamma( Vhi_a, eos_d )
            logThi_a = integrate.cumtrapz(-gammahi_a/Vhi_a,x=Vhi_a)
            logThi_a = np.append([0],logThi_a)
            logtemphi_f = interpolate.interp1d(Vhi_a,logThi_a,kind='cubic')
            indhi_a = np.where(V_a > VR)[0]
            T_a[indhi_a] = TR*np.exp(logtemphi_f(V_a[indhi_a]))

        if Vmin < VR:
            Vlo_a = np.linspace(VR,Vmin,Nsamp)
            gammalo_a = self.gamma( Vlo_a, eos_d )
            logTlo_a = integrate.cumtrapz(-gammalo_a/Vlo_a,x=Vlo_a)
            logTlo_a = np.append([0],logTlo_a)
            logtemplo_f = interpolate.interp1d(Vlo_a,logTlo_a,kind='cubic')
            indlo_a = np.where(V_a <= VR)[0]
            T_a[indlo_a] = TR*np.exp(logtemplo_f(V_a[indlo_a]))

        return T_a
#====================================================================
class FullMod(EosMod):
    """
    Abstract Equation of State class for Full Model (combines all EOS terms)
    """
    __metaclass__ = ABCMeta

    # Standard methods must be overridden (as needed) by implimentation model
    def press( self, V_a, T_a, eos_d ):
        """Returns Total Press."""
        raise NotImplementedError("'press' function not implimented for this model")

    def energy( self, V_a, T_a, eos_d ):
        """Returns Toal Energy."""
        raise NotImplementedError("'energy' function not implimented for this model")

    def therm_exp( self, V_a, T_a, eos_d ):
        """Returns Total Thermal Expansion."""
        raise NotImplementedError("'thermal_exp' function not implimented for this model")

    def bulk_mod( self, V_a, T_a, eos_d ):
        """Returns Total Bulk Modulus."""
        raise NotImplementedError("'bulk_mod' function not implimented for this model")
#====================================================================
# 3.2: Model Implementations
#====================================================================
# 3.2.1: CompressPathMod Implementations
#====================================================================
class BirchMurn3(CompressPathMod):
    def calc_press( self, V_a, eos_d ):
        V0, K0, KP0 = Control.get_params( ['V0','K0','KP0'], eos_d )

        vratio_a = V_a/V0

        press_a = 3.0/2*K0 * (vratio_a**(-7.0/3) - vratio_a**(-5.0/3)) * \
            (1 + 3.0/4*(KP0-4)*(vratio_a**(-2.0/3)-1))

        return press_a

    def calc_energy( self, V_a, eos_d ):
        V0, K0, KP0, E0 = Control.get_params( ['V0','K0','KP0','E0'], eos_d )
        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )

        vratio_a = V_a/V0

        fstrain_a = 0.5*(vratio_a**(-2.0/3) - 1)

        energy_a = E0 + 9.0/2*(V0*K0/PV_ratio)*\
            ( KP0*fstrain_a**3 + fstrain_a**2*(1-4*fstrain_a) )

        return energy_a
#====================================================================
class BirchMurn4(CompressPathMod):
    def get_param_scale_sub( self, eos_d):
        """Return scale values for each parameter"""
        V0, K0, KP0, KP20 = Control.get_params( ['V0','K0','KP0','KP20'], eos_d )
        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )

        paramkey_a = np.array(['V0','K0','KP0','KP20','E0'])
        scale_a = np.array([V0,K0,KP0,KP0/K0,K0*V0/PV_ratio])

        return scale_a, paramkey_a

    def calc_strain_energy_coeffs(self, nexp, K0, KP0, KP20 ):
        a1 = 3./2*(KP0-nexp-2)
        a2 = 3./2*(K0*KP20 + KP0*(KP0-2*nexp-3)+3+4*nexp+11./9*nexp**2)
        return a1,a2

    def calc_press( self, V_a, eos_d ):
        # globals()['set_param']( ['nexp'], [self.nexp], eos_d )
        # press_a = self.gen_finite_strain_mod.press( V_a, eos_d )
        V0, K0, KP0, KP20 = Control.get_params( ['V0','K0','KP0','KP20'], eos_d )
        nexp = +2.0

        vratio_a = V_a/V0
        fstrain_a = 1./nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self.calc_strain_energy_coeffs(nexp,K0,KP0,KP20)

        press_a = 3.0*K0*(1+a1*fstrain_a + a2*fstrain_a**2)*\
            fstrain_a*(nexp*fstrain_a+1)**((nexp+3)/nexp)
        return press_a

    def calc_energy( self, V_a, eos_d ):
        # globals()['set_param']( ['nexp'], [self.nexp], eos_d )
        # energy_a = self.gen_finite_strain_mod.energy( V_a, eos_d )
        V0, K0, KP0, KP20, E0 = Control.get_params( ['V0','K0','KP0','KP20','E0'], eos_d )
        nexp = +2.0

        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )

        vratio_a = V_a/V0
        fstrain_a = 1./nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self.calc_strain_energy_coeffs(nexp,K0,KP0,KP20)


        energy_a = E0 + 9.0*(V0*K0/PV_ratio)*\
            ( 0.5*fstrain_a**2 + a1/3*fstrain_a**3 + a2/4*fstrain_a**4)

        return energy_a
#====================================================================
class GenFiniteStrain(CompressPathMod):
    """
    Generalized Finite Strain EOS from Jeanloz1989b

    Note: nexp=2 yields Birch Murnaghan (eulerian strain) EOS
          nexp=-2 yields lagragian strain EOS
    """

    def calc_strain_energy_coeffs(self, nexp, K0, KP0, KP20=None, KP30=None):
        a1 = 3./2*(KP0-nexp-2)
        if KP20 is None:
            return a1
        else:
            a2 = 3./2*(K0*KP20 + KP0*(KP0-2*nexp-3)+3+4*nexp+11./9*nexp**2)
            if KP30 is None:
                return a1,a2
            else:
                a3 = 1./8*(9*K0**2*KP30 + 6*(6*KP0-5*nexp-6)*K0*KP20
                           +((3*KP0-5*nexp-6)**2 +10*nexp**2 + 30*nexp + 18)*KP0
                           -(50./3*nexp**3 + 70*nexp**2 + 90*nexp + 36))
                return a1,a2,a3

    def calc_press( self, V_a, eos_d ):
        V0, K0, KP0, KP20, nexp = Control.get_params( ['V0','K0','KP0','KP20','nexp'], eos_d )

        vratio_a = V_a/V0
        fstrain_a = 1./nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self.calc_strain_energy_coeffs(nexp,K0,KP0,KP20=KP20)

        press_a = 3.0*K0*(1+a1*fstrain_a + a2*fstrain_a**2)*\
            fstrain_a*(nexp*fstrain_a+1)**((nexp+3)/nexp)
        return press_a

    def calc_energy( self, V_a, eos_d ):
        V0, K0, KP0, KP20, E0, nexp = Control.get_params( ['V0','K0','KP0','KP20','E0','nexp'], eos_d )
        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )

        vratio_a = V_a/V0
        fstrain_a = 1./nexp*(vratio_a**(-nexp/3) - 1)

        a1,a2 = self.calc_strain_energy_coeffs(nexp,K0,KP0,KP20=KP20)


        energy_a = E0 + 9.0*(V0*K0/PV_ratio)*\
            ( 0.5*fstrain_a**2 + a1/3*fstrain_a**3 + a2/4*fstrain_a**4)

        return energy_a
#====================================================================
class Vinet(CompressPathMod):
    def get_param_scale_sub( self, eos_d):
        """Return scale values for each parameter"""
        V0, K0, KP0 = Control.get_params( ['V0','K0','KP0'], eos_d )
        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )

        paramkey_a = np.array(['V0','K0','KP0','E0'])
        scale_a = np.array([V0,K0,KP0,K0*V0/PV_ratio])

        return scale_a, paramkey_a

    def calc_press( self, V_a, eos_d ):
        V0, K0, KP0 = Control.get_params( ['V0','K0','KP0'], eos_d )

        eta = 3./2*(KP0-1)
        vratio_a = V_a/V0
        x_a = vratio_a**(1./3)

        press_a = 3*K0*(1-x_a)*x_a**(-2)*np.exp(eta*(1-x_a))

        return press_a

    def calc_energy( self, V_a, eos_d ):
        V0, K0, KP0, E0 = Control.get_params( ['V0','K0','KP0','E0'], eos_d )
        # print V0
        # print K0
        # print KP0
        # print E0
        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )

        eta = 3./2*(KP0-1)
        vratio_a = V_a/V0
        x_a = vratio_a**(1./3)


        energy_a = E0 + 9*K0*V0/PV_ratio/eta**2*\
            (1 + (eta*(1-x_a)-1)*np.exp(eta*(1-x_a)))

        return energy_a

    def calc_energy_perturb( self, V_a, eos_d ):
        """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""

        V0, K0, KP0, E0 = Control.get_params( ['V0','K0','KP0','E0'], eos_d )
        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )

        eta = 3./2*(KP0-1)
        vratio_a = V_a/V0
        x = vratio_a**(1./3)

        scale_a, paramkey_a = self.get_param_scale_sub( eos_d )

        # NOTE: CHECK UNITS (PV_RATIO) here
        dEdp_a = 1.0/PV_ratio*np.vstack\
            ([-3*K0*(eta**2*x*(x-1) + 3*eta*(x-1) - 3*np.exp(eta*(x-1)) + 3)\
              *np.exp(-eta*(x-1))/eta**2,
              -9*V0*(eta*(x-1) - np.exp(eta*(x-1)) + 1)*np.exp(-eta*(x-1))/eta**2,
              27*K0*V0*(2*eta*(x-1) + eta*(-x + (x-1)*(eta*(x-1) + 1) + 1)
                        -2*np.exp(eta*(x-1)) + 2)*np.exp(-eta*(x-1))/(2*eta**3),
              PV_ratio*np.ones(V_a.shape)])

        Eperturb_a = np.expand_dims(scale_a,1)*dEdp_a
        #Eperturb_a = np.expand_dims(scale_a)*dEdp_a

        return Eperturb_a, scale_a, paramkey_a
#====================================================================
class Tait(CompressPathMod):
    def __init__( self, setPmin=False, expand_adj=None):
        super(Tait, self).__init__( expand_adj=None )
        self.setPmin = setPmin
        pass
    # def __init__( self, setPmin=False, expand_adj=False ):
    #     self.setPmin = setPmin
    #     self.expand_adj = expand_adj
    #     pass

    def get_eos_params(self, eos_d):
        V0, K0, KP0 = Control.get_params( ['V0','K0','KP0'], eos_d )
        if self.setPmin:
            Pmin, = Control.get_params( ['Pmin'], eos_d )
            assert Pmin>0, 'Pmin must be positive.'
            KP20 = (KP0+1)*(KP0/K0 - 1.0/Pmin)
        else:
            KP20, = Control.get_params( ['KP20'], eos_d )

        return V0,K0,KP0,KP20

    def get_param_scale_sub( self, eos_d ):
        """Return scale values for each parameter"""

        V0, K0, KP0, KP20 = self.get_eos_params(eos_d)
        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )

        if self.setPmin:
            # [V0,K0,KP0,E0]
            paramkey_a = np.array(['V0','K0','KP0','E0'])
            scale_a = np.array([V0,K0,KP0,K0*V0/PV_ratio])
        else:
            # [V0,K0,KP0,KP20,E0]
            paramkey_a = np.array(['V0','K0','KP0','KP20','E0'])
            scale_a = np.array([V0,K0,KP0,KP0/K0,K0*V0/PV_ratio])


        return scale_a, paramkey_a

    def eos_to_abc_params(self, K0, KP0, KP20):
        a = (KP0 + 1.0)/(K0*KP20 + KP0 + 1.0)
        b = -KP20/(KP0+1.0) + KP0/K0
        c = (K0*KP20 + KP0 + 1.0)/(-K0*KP20 + KP0**2 + KP0)

        return a,b,c

    def calc_press( self, V_a, eos_d ):
        V0, K0, KP0, KP20 = self.get_eos_params(eos_d)
        a,b,c = self.eos_to_abc_params(K0,KP0,KP20)
        vratio_a = V_a/V0

        press_a = 1.0/b*(((vratio_a + a - 1.0)/a)**(-1.0/c) - 1.0)

        return press_a

    def calc_energy( self, V_a, eos_d ):
        V0, K0, KP0, KP20 = self.get_eos_params(eos_d)
        E0, = Control.get_params( ['E0'], eos_d )
        a,b,c = self.eos_to_abc_params(K0,KP0,KP20)
        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )

        vratio_a = V_a/V0

        press_a = self.calc_press( V_a, eos_d )
        eta_a = b*press_a + 1.0
        eta_pow_a = eta_a**(-c)
        #  NOTE: Need to simplify energy expression here
        energy_a = E0 + (V0/b)/PV_ratio*(a*c/(c-1)-1)\
            - (V0/b)/PV_ratio*( a*c/(c-1)*eta_a*eta_pow_a - a*eta_pow_a + a - 1)

        return energy_a

    def calc_energy_perturb_deprecate( self, V_a, eos_d ):
        """Returns Energy pertubation basis functions resulting from fractional changes to EOS params."""
        V0, K0, KP0, KP20 = self.get_eos_params(eos_d)
        E0, = Control.get_params( ['E0'], eos_d )

        a,b,c = self.eos_to_abc_params(K0,KP0,KP20)
        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )

        vratio_a = V_a/V0

        press_a = self.calc_press( V_a, eos_d )
        eta_a = b*press_a + 1.0
        eta_pow_a = eta_a**(-c)

        scale_a, paramkey_a = self.get_param_scale_sub( eos_d )

        # [V0,K0,KP0,KP20,E0]
        dEdp_a = np.ones((4, V_a.size))
        # dEdp_a[0,:] = 1.0/(PV_ratio*b*(c-1))*eta_a*(-a*eta_pow_a -1 + (1-a)*(a+c))
        dEdp_a[0,:] = 1.0/(PV_ratio*b*(c-1))*eta_a*(-a*eta_pow_a +a -1 -a*c+c) \
            + 1.0/(PV_ratio*b)*(a*c/(c-1)-1)
        dEdp_a[-1,:] = 1.0

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        # 1x3
        dEdabc_a = np.vstack\
            ([V0*eta_a/(a*b*(c-1))*(-a*eta_pow_a + a*(1-c))+c*V0/(b*(c-1)),
              V0/(b**2*(c-1))*((-a*eta_pow_a+a-1)*(c-1) + c*a*eta_a*eta_pow_a) \
              - V0/b**2*(a*c/(c-1) - 1),
              -a*V0/(b*(c-1)**2)*eta_a*eta_pow_a*(-c+(c-1)*(1-np.log(eta_a)))\
              +a*V0/(b*(c-1))*(1-c/(c-1))])
        # 3x3
        abc_jac = np.array([[-KP20*(KP0+1)/(K0*KP20+KP0+1)**2,
                             K0*KP20/(K0*KP20+KP0+1)**2,
                             -K0*(KP0+1)/(K0*KP20+KP0+1)**2],
                            [-KP0/K0**2, KP20/(KP0+1)**2 + 1./K0, -1.0/(KP0+1)],
                            [KP20*(KP0**2+2.*KP0+1)/(-K0*KP20+KP0**2+KP0)**2,
                             (-K0*KP20+KP0**2+KP0-(2*KP0+1)*(K0*KP20+KP0+1))/\
                             (-K0*KP20+KP0**2+KP0)**2,
                             K0*(KP0**2+2*KP0+1)/(-K0*KP20+KP0**2+KP0)**2]])

        dEdp_a[1:4,:] = 1.0/PV_ratio*np.dot(abc_jac.T,dEdabc_a)

        print dEdp_a.shape

        if self.setPmin:
            # [V0,K0,KP0,E0]
            print dEdp_a.shape
            dEdp_a = dEdp_a[[0,1,2,4],:]


        Eperturb_a = np.expand_dims(scale_a,1)*dEdp_a


        #Eperturb_a = np.expand_dims(scale_a)*dEdp_a

        return Eperturb_a, scale_a, paramkey_a
#====================================================================
class GenRosenfeldTaranzona(ThermalPathMod):
    """
    Generalized Rosenfeld-Taranzona Equation of State Model (Rosenfeld1998)
    - Cv takes on general form of shifted power-law as in original
    Rosenfeld-Taranzona model, but the exponent and high-temp limit are
    parameters rather than fixed
    - only applicable to isochores
    - must provide a method to evaluate properties along isochore
    """
    __metaclass__ = ABCMeta

    def get_param_scale_sub( self, eos_d):
        """Return scale values for each parameter"""
        acoef, bcoef, mexp, lognfac = Control.get_params\
            ( ['acoef','bcoef','mexp','lognfac'], eos_d )

        acoef_scl = 1.0 # This cannot be well-determined without more info
        # ...like a reference temp or energy variation
        bcoef_scl = np.abs(bcoef)
        mexp_scl = 3./5
        lognfac_scl = 0.01
        paramkey_a = np.array(['acoef','bcoef','mexp','lognfac'])
        scale_a = np.array([acoef_scl,bcoef_scl,mexp_scl,lognfac_scl])

        return scale_a, paramkey_a

    def get_param_override( self, paramkey, paramval, eos_d ):
        if paramval is None:
            paramval, = Control.get_params( [paramkey], eos_d )

        return paramval

    def calc_therm_dev( self, T_a, eos_d ):
        """
        """
        # assert False, 'calc_thermal_dev is not yet implimented'

        T0, = Control.get_params( ['T0'], eos_d )
        mexp, = Control.get_params( ['mexp'], eos_d )
        # therm_dev_a = (T_a/T0)**mexp
        therm_dev_a = (T_a/T0)**mexp - 1.0

        return therm_dev_a

    def calc_therm_dev_deriv( self, T_a, eos_d ):
        """
        """
        # assert False, 'calc_thermal_dev is not yet implimented'

        T0, = Control.get_params( ['T0'], eos_d )
        mexp, = Control.get_params( ['mexp'], eos_d )

        dtherm_dev_a = (mexp/T0)*(T_a/T0)**(mexp-1.0)

        return dtherm_dev_a


    def calc_energy( self, T_a, eos_d, acoef_a=None, bcoef_a=None ):
        """Returns Thermal Component of Energy."""
        mexp, lognfac = Control.get_params( ['mexp','lognfac'], eos_d )


        energy_pot_a = self.calc_energy_pot( T_a, eos_d, acoef_a=acoef_a,
                                            bcoef_a=bcoef_a )
        energy_kin_a = self.calc_energy_kin( T_a, eos_d )
        energy_a = energy_pot_a + energy_kin_a

        return energy_a

    def calc_energy_kin( self, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
        lognfac, = Control.get_params( ['lognfac'], eos_d )
        kB, = Control.get_consts( ['kboltz'], eos_d )
        nfac = np.exp(lognfac)
        energy_kin_a = 3.0/2*nfac*kB*T_a

        return energy_kin_a

    def calc_energy_pot( self, T_a, eos_d, acoef_a=None, bcoef_a=None ):
        """Returns Thermal Component of Energy."""
        mexp, lognfac = Control.get_params( ['mexp','lognfac'], eos_d )

        acoef_a = self.get_param_override( 'acoef', acoef_a, eos_d )
        bcoef_a = self.get_param_override( 'bcoef', bcoef_a, eos_d )

        therm_dev_a = self.calc_therm_dev( T_a, eos_d )

        energy_pot_a = acoef_a + bcoef_a*therm_dev_a

        return energy_pot_a

    def calc_heat_capacity( self, T_a, eos_d, bcoef_a=None ):
        """Calculate Heat Capacity usin."""
        heat_capacity_pot = self.calc_heat_capacity_pot( T_a, eos_d,
                                                        bcoef_a=bcoef_a )

        heat_capacity_kin = self.calc_heat_capacity_kin( T_a, eos_d )

        heat_capacity_a = heat_capacity_pot+heat_capacity_kin

        return heat_capacity_a

    def calc_heat_capacity_pot( self, T_a, eos_d, bcoef_a=None ):
        mexp, = Control.get_params( ['mexp'], eos_d )

        bcoef_a = self.get_param_override( 'bcoef', bcoef_a, eos_d )
        dtherm_dev_a = self.calc_therm_dev_deriv( T_a, eos_d )

        heat_capacity_pot_a = bcoef_a*dtherm_dev_a

        return heat_capacity_pot_a

    def calc_heat_capacity_kin( self, T_a, eos_d ):
        lognfac, = Control.get_params( ['lognfac'], eos_d )
        kB, = Control.get_consts( ['kboltz'], eos_d )
        nfac = np.exp(lognfac)
        heat_capacity_kin_a = + 3.0/2*nfac*kB

        return heat_capacity_kin_a

    def calc_entropy_pot( self, T_a, eos_d, Tref=None, bcoef_a=None ):
        mexp, = Control.get_params( ['mexp'], eos_d )

        Tref = self.get_param_override( 'T0', Tref, eos_d )

        Cv_pot = self.calc_heat_capacity_pot( T_a, eos_d, bcoef_a=bcoef_a )
        Cv_ref_pot = self.calc_heat_capacity_pot( Tref, eos_d, bcoef_a=bcoef_a )
        dSpot_a = (Cv_pot-Cv_ref_pot)/(mexp-1.0)

        return dSpot_a

    def calc_entropy_kin( self, T_a, eos_d, Tref=None ):
        Tref = self.get_param_override( 'T0', Tref, eos_d )

        Cv_kin = self.calc_heat_capacity_kin( T_a, eos_d )
        dSkin_a = Cv_kin*np.log( T_a/Tref )

        return dSkin_a

    def calc_entropy( self, T_a, eos_d, Tref=None, Sref=None, bcoef_a=None ):
        """Returns Entropy."""
        mexp, = Control.get_params( ['mexp'], eos_d )

        Sref = self.get_param_override( 'S0', Sref, eos_d )
        Tref = self.get_param_override( 'T0', Tref, eos_d )

        Cv_pot = self.calc_heat_capacity_pot( T_a, eos_d, bcoef_a=bcoef_a )
        Cv_ref_pot = self.calc_heat_capacity_pot( Tref, eos_d, bcoef_a=bcoef_a )
        delS_pot = (Cv_pot-Cv_ref_pot)/(mexp-1.0)

        Cv_kin = self.calc_heat_capacity_kin( T_a, eos_d )
        delS_kin = Cv_kin*np.log( T_a/Tref )

        entropy_a = Sref + delS_pot + delS_kin
        return entropy_a
#====================================================================
class RosenfeldTaranzonaCompress(ThermalMod):
    """
    Volume-dependent Rosenfeld-Taranzona Equation of State
      - must impliment particular volume-dependence

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def calc_RT_coef( self, V_a, eos_d ):
        """
        Must implement a method for determining RT coefficients as a function of vol
        """
        acoef_a = []
        bcoef_a = []
        return acoef_a, bcoef_a

    def calc_RT_coef_deriv( self, V_a, eos_d ):
        V0, = Control.get_params( ['V0'], eos_d )

        # Use numerical deriv
        dV = V0*1e-5
        acoef_a, bcoef_a = self.calc_RT_coef( V_a, eos_d )
        acoef_hi_a, bcoef_hi_a = self.calc_RT_coef( V_a+dV, eos_d )

        acoef_deriv_a = (acoef_hi_a-acoef_a)/dV
        bcoef_deriv_a = (bcoef_hi_a-bcoef_a)/dV

        return acoef_deriv_a, bcoef_deriv_a

    def calc_energy( self, V_a, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
        acoef_a, bcoef_a = self.calc_RT_coef( V_a, eos_d )
        energy_a = GenRosenfeldTaranzona().calc_energy\
            ( T_a, eos_d, acoef_a=acoef_a, bcoef_a=bcoef_a )
        return energy_a

    def calc_energy_pot( self, V_a, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
        acoef_a, bcoef_a = self.calc_RT_coef( V_a, eos_d )
        energy_pot_a = GenRosenfeldTaranzona().calc_energy_pot\
            ( T_a, eos_d, acoef_a=acoef_a, bcoef_a=bcoef_a )
        return energy_pot_a

    def calc_energy_kin( self, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
        energy_kin_a = GenRosenfeldTaranzona().calc_energy_kin( T_a, eos_d )
        return energy_kin_a

    def calc_heat_capacity( self, V_a, T_a, eos_d ):
        """Calculate Heat Capacity usin."""
        acoef_a, bcoef_a = self.calc_RT_coef( V_a, eos_d )
        heat_capacity_a = GenRosenfeldTaranzona().calc_heat_capacity\
            ( T_a, eos_d, bcoef_a=bcoef_a )
        return heat_capacity_a

    def calc_entropy( self, V_a, T_a, eos_d ):
        V0, S0 = Control.get_params( ['V0','S0'], eos_d )
        T0, = Control.get_params( ['T0'], eos_d )

        acoef_a, bcoef_a = self.calc_RT_coef( V_a, eos_d )
        gamma_mod = eos_d['modtype_d']['GammaMod']

        Tref_a = gamma_mod.temp( V_a, T0, eos_d )

        S_a = GenRosenfeldTaranzona().calc_entropy( T_a, eos_d, Tref=Tref_a,
                                                   Sref=S0, bcoef_a=bcoef_a )
        return S_a
#====================================================================
class RosenfeldTaranzonaPoly(RosenfeldTaranzonaCompress):
    """
    Polynomial volume-dependence for Rosenfeld-Taranzona Equation of State
    - First implemented by  (Saika-Voivod2000)
    """
    __metaclass__ = ABCMeta

    def calc_RT_coef( self, V_a, eos_d ):
        """
        Must implement a method for determining RT coefficients as a function of vol
        """
        poly_acoef_a = Control.get_array_params( 'acoef', eos_d )
        poly_bcoef_a = Control.get_array_params( 'bcoef', eos_d )

        # NOTE: Python's order of polynomial coefficients (for numpy) is
        #       highest power first, OPPOSITE of how we store it.
        #  So array must be reversed prior to evaluation
        acoef_a = np.polyval(poly_acoef_a[::-1], V_a)
        bcoef_a = np.polyval(poly_bcoef_a[::-1], V_a)
        return acoef_a, bcoef_a

    def calc_press( self, V_a, T_a, eos_d ):
        T0,V0,mexp,lognfac = Control.get_params( ['T0','V0','mexp','lognfac'], eos_d )

        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )
        compress_path_mod = eos_d['modtype_d']['CompressPathMod']

        nfac = np.exp(lognfac)
        assert mexp==3./5, 'currently hardcoded to mexp=3/5'
        # assert nfac==1, 'currently hardcoded to nfac=1'

        P_ref_a = compress_path_mod.calc_press( V_a, eos_d )

        dV = V0*1e-4
        acoef_a, bcoef_a = self.calc_RT_coef( V_a, eos_d )
        acoef_hi_a, bcoef_hi_a = self.calc_RT_coef( V_a+dV, eos_d )

        dadV_a = (acoef_hi_a-acoef_a)/dV
        dbdV_a = (bcoef_hi_a-bcoef_a)/dV



        # Equation 6 (Spera2011)

        # NOTE that the temperature factor has been removed

        # NOTE: Check PV_ratio
        press_therm_a = P_ref_a \
            + 1.0/PV_ratio*( (T_a/T0 - 1)*dadV_a \
                        + 5./2*T_a**(3./5)*((T_a/T0)**(2./5) - 1.0)*dbdV_a)
        # press_therm_a = T_a/T0*P_ref_a \
        #     + 1.0/PV_ratio*( (T_a/T0 - 1)*dadV_a \
        #                 + 5./2*T_a**(3./5)*((T_a/T0)**(2./5) - 1.0)*dbdV_a)

        return press_therm_a
#====================================================================
class RosenfeldTaranzonaPerturb(RosenfeldTaranzonaCompress):
    """
    Vol-dependence for Rosenfeld-Taranzona EOS from Compress Path Perturbation
    """
    __metaclass__ = ABCMeta

    @classmethod
    def hello_world(cls):
        print 'hello world'
        pass

    def get_thermal_perturb_param( self, eos_d ):
        compress_path_mod = eos_d['modtype_d']['CompressPathMod']
        compress_param_scale_a, param_list_a =  compress_path_mod.get_param_scale( eos_d )

        paramkey_a = []
        for param in param_list_a:
            paramkey_a.append( 'd'+param+'th' )

        paramkey_a = np.array(paramkey_a)
        return paramkey_a

    def calc_therm_dev( self, V_a, T_a, eos_d ):
        """
        Not yet implimented!

        """
        # assert False, 'calc_thermal_dev is not yet implimented'

        T0, = Control.get_params( ['T0'], eos_d )
        gamma_mod = eos_d['modtype_d']['GammaMod']
        T_ref_a = gamma_mod.temp( V_a, T0, eos_d )

        mexp, = Control.get_params( ['mexp'], eos_d )
        therm_dev_a = (T_a/T0)**mexp - (T_ref_a/T0)**mexp

        return therm_dev_a

    def get_param_scale_sub( self, eos_d):
        """Return scale values for each parameter"""
        paramkey_a = self.get_thermal_perturb_param( eos_d )

        scale_a = np.ones(paramkey_a.size)

        T0, = Control.get_params( ['T0'], eos_d )
        T0_scl = T0
        mexp_scl = 3./5
        lognfac_scl = 0.01

        paramkey_a = np.append(paramkey_a,['T0','mexp','lognfac'])
        scale_a = np.append(scale_a,[T0_scl,mexp_scl,lognfac_scl])

        return scale_a, paramkey_a

    def calc_RT_coef( self, V_a, eos_d ):
        """
        Must implement a method for determining RT coefficients as a function of vol
        """

        # paramkey_a = self.get_thermal_perturb_param( eos_d )
        compress_path_mod = eos_d['modtype_d']['CompressPathMod']
        dE0th,dV0th,dK0th, dKP0th = Control.get_params( ['dE0th','dV0th','dK0th',
                                                         'dKP0th'], eos_d )

        T0,mexp = Control.get_params( ['T0','mexp'], eos_d )


        compress_path_mod = eos_d['modtype_d']['CompressPathMod']
        expand_adj = compress_path_mod.expand_adj
        # NOTE: must use energy_perturb, rather than calc_energy_perturb!!!

        # Eperturb_a, compress_scale_a, compress_paramkey_a = \
        #     compress_path_mod.calc_energy_perturb( V_a, eos_d,
        #                                           apply_expand_adj=expand_adj)

        Eperturb_a, compress_scale_a, compress_paramkey_a = compress_path_mod.energy_perturb(V_a, eos_d)
        # Eperturb_a = compress_path_mod.calc_energy_perturb( V_a, eos_d )[0]
        # compress_scale_a, compress_paramkey_a = \
        #     compress_path_mod.get_param_scale( eos_d, apply_expand_adj=True )

        E0scl = compress_scale_a[compress_paramkey_a=='E0']
        V0scl = compress_scale_a[compress_paramkey_a=='V0']
        K0scl = compress_scale_a[compress_paramkey_a=='K0']
        KP0scl = compress_scale_a[compress_paramkey_a=='KP0']

        bcoef_a = 1.0/T0**mexp*\
            ( dE0th*E0scl*Eperturb_a[compress_paramkey_a=='E0',:]
             + dV0th*V0scl*Eperturb_a[compress_paramkey_a=='V0',:]
             + dK0th*K0scl*Eperturb_a[compress_paramkey_a=='K0',:]
             + dKP0th*KP0scl*Eperturb_a[compress_paramkey_a=='KP0',:])

        # NOTE: ensure that KP20 is included if either 4th order
        #      compressPathMod or expand_adj used

        # if compress_path_mod.expand_adj:
        if np.any(compress_paramkey_a=='KP20'):
            dKP20th, = Control.get_params( ['dKP20th'], eos_d )
            KP20scl = compress_scale_a[compress_paramkey_a=='KP20']

            bcoef_a += 1.0/T0**mexp*dKP20th*KP20scl\
                *Eperturb_a[compress_paramkey_a=='KP20',:]


        # acoef_a = compress_path_mod.calc_energy( V_a, eos_d )
        acoef_a = np.zeros( V_a.size )

        return acoef_a, bcoef_a

    def calc_energy_pot_diff( self, V_a, T_a, eos_d ):
        # T0, = Control.get_params( ['T0'], eos_d )

        # gamma_mod = eos_d['modtype_d']['GammaMod']
        # T_ref_a = gamma_mod.temp( V_a, T0, eos_d )

        # del_energy_pot_a = self.calc_energy_pot( V_a, T_a, eos_d ) \
        #     - self.calc_energy_pot( V_a, T_ref_a, eos_d )

        therm_dev_a = self.calc_therm_dev( V_a, T_a, eos_d )

        acoef_a, bcoef_a = self.calc_RT_coef( V_a, eos_d )

        del_energy_pot_a = bcoef_a*therm_dev_a
        return del_energy_pot_a

    def calc_energy_kin_diff( self, V_a, T_a, eos_d ):
        T0, = Control.get_params( ['T0'], eos_d )

        gamma_mod = eos_d['modtype_d']['GammaMod']
        T_ref_a = gamma_mod.temp( V_a, T0, eos_d )

        del_energy_kin_a = self.calc_energy_kin( T_a, eos_d ) \
            - self.calc_energy_kin( T_ref_a, eos_d )

        return del_energy_kin_a

    def calc_energy( self, V_a, T_a, eos_d ):
        dE_pot = self.calc_energy_pot_diff( V_a, T_a, eos_d )
        dE_kin = self.calc_energy_kin_diff( V_a, T_a, eos_d )
        dE_tot = dE_pot + dE_kin

        return dE_tot

    def calc_free_energy( self, V_a, T_a, eos_d ):
        dE_tot = self.calc_energy( V_a, T_a, eos_d )
        S_tot = self.calc_entropy( V_a, T_a, eos_d )
        dF_tot = dE_tot - T_a*S_tot

        return dF_tot

    def calc_press( self, V_a, T_a, eos_d ):
        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )
        V0, = Control.get_params( ['V0'], eos_d )

        # Use numerical deriv
        dV = V0*1e-5
        F_a = self.calc_free_energy( V_a, T_a, eos_d )
        F_hi_a = self.calc_free_energy( V_a+dV, T_a, eos_d )
        press_therm_a = -PV_ratio*(F_hi_a-F_a)/dV

        return press_therm_a
#====================================================================
class MieGrun(ThermalMod):
    """
    Mie-Gruneisen Equation of State Model
    (requires extension to define thermal energy model)
    """
    __metaclass__ = ABCMeta

    def press( self, V_a, T_a, eos_d ):
        V_a, T_a = fill_array( V_a, T_a )

        PV_ratio, = Control.get_consts( ['PV_ratio'], eos_d )
        gamma_mod, = Control.get_modtypes( ['GammaMod'], eos_d )

        # Needed functions
        energy_therm_a = self.energy( V_a, T_a, eos_d )
        gamma_a = gamma_mod.gamma( V_a, eos_d )

        press_therm_a = PV_ratio*(gamma_a/V_a)*energy_therm_a
        return press_therm_a

    @abstractmethod
    def energy( self, V_a, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
#====================================================================
class MieGrunDebye(MieGrun):
    def __init__( self ):
       super(MieGrunDebye, self).__init__( path_const='V' )

    def energy( self, V_a, T_a, eos_d ):
        '''
        Thermal Energy for Debye model

        Relies on reference profile properties stored in eos_d defined by:
        * debye_temp_f( V_a, T_a )
        * ref_temp_f( V_a, T_a )
        '''
        V_a, T_a = fill_array( V_a, T_a )

        # NOTE: T0 refers to temp on ref adiabat evaluated at V0
        Cvmax, T0, thetaR = Control.get_params( ['Cvmax','T0','thetaR'], eos_d )
        TS_ratio, = Control.get_consts( ['TS_ratio'], eos_d )
        gamma_mod, = Control.get_modtypes( ['GammaMod'], eos_d )

        theta_a = gamma_mod.temp( V_a, thetaR, eos_d )
        Tref_a = gamma_mod.temp( V_a, T0, eos_d )
        # print theta_a

        ######################
        # NOTE: Some weird issue with TS_ratio!!!
        ######################
        # energy_therm_a = (Cvmax/TS_ratio)*(
        #     + T_a*self.debye_func( theta_a/T_a )
        #     - Tref_a*self.debye_func( theta_a/Tref_a ) )
        energy_therm_a = (Cvmax)*(
            + T_a*self.debye_func( theta_a/T_a )
            - Tref_a*self.debye_func( theta_a/Tref_a ) )

        return energy_therm_a

    def entropy( self, V_a, T_a, eos_d ):
        V_a, T_a = fill_array( V_a, T_a )

        Cvmax, thetaR = Control.get_params( ['Cvmax','thetaR'], eos_d )
        TS_ratio, = Control.get_consts( ['TS_ratio'], eos_d )
        gamma_mod, = Control.get_modtypes( ['GammaMod'], eos_d )

        theta_a = gamma_mod.temp( V_a, thetaR, eos_d )
        x_a = theta_a/T_a

        # entropy_a = Cvmax*Cv_const/3. \
            #     *(4*debye_func( x_a )-3*np.log( 1-np.exp( -x_a ) ) )
        entropy_a = 1.0/3*(Cvmax/TS_ratio) \
            *(4*debye_func( x_a )-3*np.log( np.exp( x_a ) - 1 ) + 3*x_a)

        return entropy_a

    def heat_capacity( self, V_a, T_a, eos_d ):
        V_a, T_a = fill_array( V_a, T_a )

        Cvmax, thetaR = Control.get_params( ['Cvmax','thetaR'], eos_d )
        TS_ratio, = Control.get_consts( ['TS_ratio'], eos_d )
        gamma_mod, = Control.get_modtypes( ['GammaMod'], eos_d )

        theta_a = gamma_mod.temp( V_a, thetaR, eos_d )

        # The reference adiabat terms in the internal energy are temperature
        # independent, and thus play no role in heat capacity
        x_a = theta_a/T_a
        # heat_capacity_a = (Cvmax/TS_ratio)*\
        #     (4*self.debye_func( x_a )-3*x_a/(np.exp(x_a)-1))

        ######################
        # NOTE: Some weird issue with TS_ratio!!!
        ######################
        heat_capacity_a = (Cvmax)*\
            (4*self.debye_func( x_a )-3*x_a/(np.exp(x_a)-1))

        return heat_capacity_a

    def debye_func( self, x_a ):
        """
        Return debye integral value

        - calculation done using interpolation in a lookup table
        - interpolation done in log-space where behavior is close to linear
        - linear extrapolation is implemented manually
        """

        if np.isscalar( x_a ):
            assert x_a >= 0, 'x_a values must be greater than zero.'
        else:
            assert all( x_a >= 0 ), 'x_a values must be greater than zero.'
            # Lookup table
            # interpolate in log space where behavior is nearly linear
            debyex_a = np.array( [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                  1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8,
                                  3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0,
                                  5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0] )
            debyelogf_a = np.array( [ 0.0, -0.03770187, -0.07580279, -0.11429475,
                                     -0.15316866, -0.19241674, -0.2320279 , -0.27199378,
                                     -0.31230405, -0.35294619, -0.39390815, -0.43518026,
                                     -0.47674953, -0.51860413, -0.56072866, -0.64573892,
                                     -0.73167389, -0.81841793, -0.90586032, -0.99388207,
                                     -1.08236598, -1.17119911, -1.26026101, -1.34944183,
                                     -1.43863241, -1.52771969, -1.61660856, -1.70519469,
                                     -1.79338479, -1.88108917, -1.96822938, -2.05471771,
                                     -2.14049175, -2.35134476, -2.55643273, -2.75507892,
                                     -2.94682783, -3.13143746, -3.30880053, -3.47894273,
                                     -3.64199587, -3.79820337, -3.94785746] )
            # Create interpolation function
            logdeb_func = interpolate.interp1d( debyex_a, debyelogf_a,
                                               kind='cubic',
                                               bounds_error=False,
                                               fill_value=np.nan )
            logfval_a = logdeb_func( x_a )

            # Check for extrapolated values indicated by NaN
            #   - replace with linear extrapolation
            logfextrap_a = debyelogf_a[-1] + (x_a - debyex_a[-1]) \
                *(debyelogf_a[-1]-debyelogf_a[-2])\
                /(debyex_a[-1]-debyex_a[-2])
            logfval_a = np.where( x_a > debyex_a[-1], logfextrap_a,
                                 logfval_a )
            # exponentiate to get integral value
            return np.exp( logfval_a )
#====================================================================
class GammaPowLaw(GammaMod):
    def __init__( self, V0ref=False ):
        self.V0ref = V0ref
        pass

    def get_param_scale_sub( self, eos_d ):
        """Return scale values for each parameter"""


        gammaR, qR = Control.get_params( ['gammaR','qR'], eos_d )

        if self.V0ref:
            VR, = Control.get_params( ['V0'], eos_d )
            VRkey = 'V0'
        else:
            VR, = Control.get_params( ['VR'], eos_d )
            VRkey = 'VR'

        gammaR_scl = 1.0
        qR_scl = 1.0
        VR_scl = VR

        paramkey_a = np.array(['gammaR','qR',VRkey])
        scale_a = np.array([gammaR,qR,VR])

        return scale_a, paramkey_a

    def gamma( self, V_a, eos_d ):
        # OLD version fixed to zero-press ref volume
        # V0, gamma0, qR = Control.get_params( ['V0','gamma0','qR'], eos_d )
        # gamma_a = gamma0 *(V_a/V0)**qR

        # generalized version
        gammaR, qR = Control.get_params( ['gammaR','qR'], eos_d )

        if self.V0ref:
            VR, = Control.get_params( ['V0'], eos_d )
        else:
            VR, = Control.get_params( ['VR'], eos_d )


        gamma_a = gammaR *(V_a/VR)**qR

        return gamma_a

    def temp( self, V_a, TR, eos_d ):
        """
        Return temperature for debye model
        V_a: sample volume array
        TR: temperature at V=VR
        """

        # OLD version fixed to zero-press ref volume
        gammaR, qR = Control.get_params( ['gammaR','qR'], eos_d )
        gamma_a = self.gamma( V_a, eos_d )
        T_a = TR*np.exp( -(gamma_a - gammaR)/qR )

        return T_a
#====================================================================
class GammaFiniteStrain(GammaMod):
    def __init__( self, V0ref=False ):
        self.V0ref = V0ref
        pass

    def get_param_scale_sub( self, eos_d ):
        """Return scale values for each parameter"""


        # gammaR, qR = Control.get_params( ['gammaR','qR'], eos_d )
        gammaR, gammapR = Control.get_params( ['gammaR','gammapR'], eos_d )

        if self.V0ref:
            VR, = Control.get_params( ['V0'], eos_d )
            VRkey = 'V0'
        else:
            VR, = Control.get_params( ['VR'], eos_d )
            VRkey = 'VR'

        gammaR_scl = 1.0
        gammapR_scl = 1.0
        #qR_scl = 1.0
        VR_scl = VR

        # paramkey_a = np.array(['gammaR','qR',VRkey])
        # scale_a = np.array([gammaR,qR,VR])
        paramkey_a = np.array(['gammaR','gammapR',VRkey])
        scale_a = np.array([gammaR,gammapR,VR])

        return scale_a, paramkey_a

    def gamma( self, V_a, eos_d ):
        # OLD version fixed to zero-press ref volume
        # V0, gamma0, qR = Control.get_params( ['V0','gamma0','qR'], eos_d )
        # gamma_a = gamma0 *(V_a/V0)**qR

        # generalized version
        # gammaR, qR = Control.get_params( ['gammaR','qR'], eos_d )
        gammaR, gammapR = Control.get_params( ['gammaR','gammapR'], eos_d )
        a1 = 6*gammaR
        a2 = -12*gammaR +36*gammaR**2 -18*gammapR

        if self.V0ref:
            VR, = Control.get_params( ['V0'], eos_d )
        else:
            VR, = Control.get_params( ['VR'], eos_d )


        # gamma_a = gammaR *(V_a/VR)**qR
        fstr = 0.5*((VR/V_a)**(2./3)-1.0)

        gamma_a = (2*fstr+1)*(a1+a2*fstr)/(6*(1+a1*fstr+0.5*a2*fstr**2))

        return gamma_a
#====================================================================
class ThermalPressMod(FullMod):
    def press( self, V_a, T_a, eos_d ):
        """Returns Press variation along compression curve."""
        V_a, T_a = fill_array( V_a, T_a )
        compress_path_mod, thermal_mod = Control.get_modtypes( ['CompressPathMod', 'ThermalMod'],
                                               eos_d )
        press_a = np.squeeze( compress_path_mod.press( V_a, eos_d )
                             + thermal_mod.press( V_a, T_a, eos_d ) )
        return press_a

    def energy( self, V_a, T_a, eos_d ):
        """Returns Thermal Component of Energy."""
        V_a, T_a = fill_array( V_a, T_a )
        compress_path_mod, thermal_mod = Control.get_modtypes( ['CompressPathMod', 'ThermalMod'],
                                               eos_d )
        energy_a = np.squeeze( compress_path_mod.energy( V_a, eos_d )
                              + thermal_mod.energy( V_a, T_a, eos_d ) )

        return energy_a

    def free_energy( self, V_a, T_a, eos_d ):
        """Returns Free Energy."""
        V_a, T_a = fill_array( V_a, T_a )
        compress_path_mod, thermal_mod = Control.get_modtypes( ['CompressPathMod', 'ThermalMod'],
                                               eos_d )
        dFtot_a = thermal_mod.calc_free_energy( V_a, T_a, eos_d )
        Eref_a = compress_path_mod.energy( V_a, eos_d )
        free_energy_a = np.squeeze( Eref_a + dFtot_a )
        return free_energy_a

    def entropy( self, V_a, T_a, eos_d ):
        """Returns Free Energy."""
        V_a, T_a = fill_array( V_a, T_a )
        thermal_mod, = Control.get_modtypes( ['ThermalMod'], eos_d )
        S_a = np.squeeze( thermal_mod.entropy( V_a, T_a, eos_d ) )

        return S_a

    def heat_capacity( self, V_a, T_a, eos_d ):
        """Returns Free Energy."""
        V_a, T_a = fill_array( V_a, T_a )
        thermal_mod, = Control.get_modtypes( ['ThermalMod'], eos_d )
        Cv_a = np.squeeze( thermal_mod.heat_capacity( V_a, T_a, eos_d ) )

        return Cv_a
#====================================================================
