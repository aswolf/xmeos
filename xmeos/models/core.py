# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass

import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod

# xmeos.models.Calculator
# xmeos.models.Eos
#====================================================================
# EOS Model Classes
#====================================================================
class EosMod(with_metaclass(ABCMeta)):
    """
    Abstract Equation of State Parent Base class
    """

    def __init__( self ):
        self._init_calculators()
        pass

    ######################
    # Calculator methods #
    ######################
    def _init_calculators( self ):
        self._compress_calculator = None
        self._thermal_calculator = None
        self._gamma_calculator = None
        self._heat_capacity_calculator = None
        self._thermal_exp_calculator = None
        pass

    def _get_calculators( self ):
        """
        Get list of valid Eos calculators
        """
        calculators = {}

        if self._compress_calculator:
            calculators.update(self._compress_calculator.param_names())

        if self._thermal_calculator:
            calculators.update(self._thermal_calculator.param_names())

        if self._gamma_calculator:
            calculators.update(self._gamma_calculator.param_names())

        if self._heat_capacity_calculator:
            calculators.update(self._heat_capacity_calculator.param_names())

        if self._thermal_exp_calculator:
            calculators.update(self._thermal_exp_calculator.param_names())

        return calculators

    def _set_calculator( self, calc, kind='compress' ):
        """
        Store calculator instance in its correct place.

        Parameters
        ----------
        calc : Calculator obj
            Instance of Calculator().
        kind : {'compress','thermal','gamma','heat_capacity','thermal_exp'}
            Kind of calculator object.

        """
        assert isinstance(Calculator(), calc), \
            'calc must be a valid Calculator object instance.'

        if   kind=='compress':
            self._compress_calculator = calc
        elif kind=='thermal':
            self._thermal_calculator = calc
        elif kind=='gamma':
            self._gamma_calculator = calc
        elif kind=='heat_capacity':
            self._heat_capacity_calculator = calc
        elif kind=='thermal_exp':
            self._thermal_exp_calculator = calc
        else:
            raise NotImplementedError(kind+' is not a supported '+\
                                      'calculator kind.')
        pass

    @property
    def compress_calculator( self ):
        """
        Calculator implimenting compression properties.

        """
        return self._compress_calculator

    @property
    def thermal_calculator( self ):
        """
        Calculator implimenting thermal properties (e.g. thermal press).

        """
        return self._thermal_calculator

    @property
    def gamma_calculator( self ):
        """
        Calculator implimenting (reference) Grüneisen gamma profile.

        """
        return self._gamma_calculator

    @property
    def heat_capacity_calculator( self ):
        """
        Calculator implimenting heat capacity/thermal energy model.

        """
        return self._heat_capacity_calculator

    @property
    def thermal_exp_calculator( self ):
        """
        Calculator implimenting thermal expansion properties.

        """
        return self._thermal_exp_calculator

    #####################
    # Parameter methods #
    #####################
    def _get_required_param_names( self ):
        """
        Get param names from calculators.

        """
        param_names = {}

        for calc in self._get_calculators():
            param_names.update(calc.param_names())

        return param_names

    def _validate_param_names( self, param_names ):
        if type(param_names) == str:
            param_names = [param_names]

        assert all( name in self._param_names for name in param_names ),\
            'All provided param_names must be valid parameter names.'

        return param_names

    def get_param_names( self ):
        """
        Get array of parameter names.

        """
        return self._param_names

    def get_param_units( self, param_names=[], all_params=False ):
        """
        Get units for listed parameters.

        Parameters
        ----------
        param_names : str array
            list of parameter names
        all_params : bool, default False
            if true, returns units for all parameters

        Returns
        -------
        units : double array
            list of units for selected parameters

        """
        if all_params:
            param_names = self._get_param_names()

        param_names = self._validate_param_names( param_names )

        units = []
        for key in param_names:
            unit = self._param_units[key]
            units.append(unit)

        return units

    def get_param_values( self, param_names=[], all_params=False ):
        """
        Get values for listed parameters.

        Parameters
        ----------
        param_names : str array
            list of parameter names
        all_params : bool, default False
            if true, returns units for all parameters

        Returns
        -------
        values : double array
            list of values for selected parameters

        """
        if all_params:
            param_names = self._get_param_names()

        param_names = self._validate_param_names( param_names )

        values = []
        for key in param_names:
            value = self._param_values[key]
            values.append(value)

        return values

    def set_param_values( self, param_names=[], param_values=[]  ):
        """
        Set new values for listed parameters.

        Parameters
        ----------
        param_names : str array
            list of parameter names
        param_values : double array
            list of parameter values

        """
        assert len(param_names)==len(param_values), \
            'param_names and param_values must have the same length'

        for name, value in zip(param_names, param_values):
            self._param_values(name,value)

        pass

    def get_param_scale( self, param_names=[], all_params=False ):
        """
        Get scale values for listed parameters.

        Parameters
        ----------
        param_names : str array
            list of parameter names
        all_params : bool, default False
            if true, returns units for all parameters

        Returns
        -------
        values : double array
            list of parameter scale values for selected parameters

        """
        if all_params:
            param_names = self._get_param_names()

        param_names = self._validate_param_names( param_names )

        scales = []
        for key in param_names:
            scale = self._param_scales[key]
            scales.append(scale)

        return scales

    # def get_param_scale_sub( self, eos_d ):
    #     raise NotImplementedError("'get_param_scale_sub' function not implimented for this model")
#====================================================================
class CompressMod(with_metaclass(ABCMeta, EosMod)):
    """
    Abstract Equation of State class for a reference Compression Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    _path_opts = ['T','S']
    _kind_opts = ['Vinet','BirchMurn3','BirchMurn4','GenFiniteStrain','Tait']

    def __init__( self, kind='Vinet',path_const='T', level_const=300 ):
        self._init_calculator(kind)
        pass

    def _init_calculator( self, kind, path_const, level_cosnt ):
        assert kind in self._kind_opts, kind + ' is not a valid ' + \
            'CompressMod Calculator. You must select one of: ' + self._kind_opts

        assert path_const in self._path_opts, path_const + ' is not a valid ' + \
            'path const. You must select one of: ' + self._path_opts

        self._kind = kind
        self._path_const = path_const
        self._level_const = level_const

        _kind_opts = ['Vinet','BirchMurn3','BirchMurn4','GenFiniteStrain','Tait']
        if   kind=='Vinet':
            calc = compress.Vinet(path_const=path_const,
                                  level_const=level_const)
        elif kind=='BirchMurn3':
            calc = compress.BirchMurn3(path_const=path_const,
                                       level_const=level_const)
        elif kind=='BirchMurn4':
            calc = compress.BirchMurn4(path_const=path_const,
                                       level_const=level_const)
        elif kind=='GenFiniteStrain':
            calc = compress.GenFiniteStrain(path_const=path_const,
                                            level_const=level_const)
        elif kind=='Tait':
            calc = compress.Tait(path_const=path_const,
                                 level_const=level_const)
        else:
            raise NotImplementedError(kind+' is not a valid '+\
                                      'CompressMod Calculator.')

        self._set_calculator( calc, kind='compress' )
        pass

    @property
    def path_const(self):
        return self._path_const

    @property
    def level_const(self):
        return self._level_const

    def press( self, V_a, eos_d, apply_expand_adj=True):
        if self.supress_press:
            zero_a = 0*V_a
            return zero_a

        else:
            press_a = self._calc_press(V_a, eos_d)
            if self.expand_adj and apply_expand_adj:
                ind_exp = self.get_ind_expand(V_a, eos_d)
                if (ind_exp.size>0):
                    press_a[ind_exp] = self.expand_adj_mod._calc_press( V_a[ind_exp], eos_d )

            return press_a
        pass

    def energy( self, V_a, eos_d, apply_expand_adj=True ):
        if self.supress_energy:
            zero_a = 0*V_a
            return zero_a

        else:
            energy_a =  self._calc_energy(V_a, eos_d)
            if self.expand_adj and apply_expand_adj:
                ind_exp = self.get_ind_expand(V_a, eos_d)
                if apply_expand_adj and (ind_exp.size>0):
                    energy_a[ind_exp] = self.expand_adj_mod._calc_energy( V_a[ind_exp], eos_d )

            return energy_a

    def bulk_mod( self, V_a, eos_d, apply_expand_adj=True ):
        bulk_mod_a =  self._calc_bulk_mod(V_a, eos_d)
        if self.expand_adj and apply_expand_adj:
            ind_exp = self.get_ind_expand(V_a, eos_d)
            if apply_expand_adj and (ind_exp.size>0):
                bulk_mod_a[ind_exp] = self.expand_adj_mod._calc_bulk_mod( V_a[ind_exp], eos_d )

        return bulk_mod_a

    def bulk_mod_deriv(  self,V_a, eos_d, apply_expand_adj=True ):
        bulk_mod_deriv_a =  self._calc_bulk_mod_deriv(V_a, eos_d)
        if self.expand_adj and apply_expand_adj:
            ind_exp = self.get_ind_expand(V_a, eos_d)
            if apply_expand_adj and (ind_exp.size>0):
                bulk_mod_deriv_a[ind_exp] = self.expand_adj_mod_deriv._calc_bulk_mod_deriv( V_a[ind_exp], eos_d )

        return bulk_mod_deriv_a

    def energy_perturb( self, V_a, eos_d, apply_expand_adj=True ):
        # Eval positive press values
        Eperturb_pos_a, scale_a, paramkey_a  = self._calc_energy_perturb( V_a, eos_d )

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
            ind_exp = self.get_ind_expand(V_a, eos_d)
            if ind_exp.size>0:
                Eperturb_adj_a = \
                    self.expand_adj_mod._calc_energy_perturb( V_a[ind_exp],
                                                            eos_d )[0]
                Eperturb_a[:,ind_exp] = Eperturb_adj_a

            return Eperturb_a, scale_a, paramkey_a

    #   Standard methods must be overridden (as needed) by implimentation model

    def get_param_scale_sub( self, eos_d):
        raise NotImplementedError("'get_param_scale_sub' function not implimented for this model")

    ####################
    # Required Methods #
    ####################
    @abstractmethod
    def _calc_press( self, V_a, eos_d ):
        """Returns Press variation along compression curve."""
        pass

    @abstractmethod
    def _calc_energy( self, V_a, eos_d ):
        """Returns Energy along compression curve."""
        pass

    ####################
    # Optional Methods #
    ####################
    def _calc_energy_perturb( self, V_a, eos_d ):
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

    def _calc_bulk_mod( self, V_a, eos_d ):
        """Returns Bulk Modulus variation along compression curve."""
        raise NotImplementedError("'bulk_mod' function not implimented for this model")

    def _calc_bulk_mod_deriv( self, V_a, eos_d ):
        """Returns Bulk Modulus Deriv (K') variation along compression curve."""
        raise NotImplementedError("'bulk_mod_deriv' function not implimented for this model")
#====================================================================

#====================================================================
class Calculator(with_metaclass(ABCMeta)):
    """
    Abstract Equation of State Calculator

    """

    def __init__( self ):
        self._param_names = None
        self._required_calculators = None
        pass

    @property
    def param_names(self):
        """
        List of parameter names for this Eos Calculator.

        """
        return self._param_names

    @property
    def required_calculators(self):
        """
        List of required coupled calculator types.

        This Eos Calculator requires other calculators to function. This
        provides a list of those calculator types.

        """
        return self._param_names
#====================================================================

#====================================================================
def init_consts( eos_d ):
    eos_d['const_d'] = default_consts()
    pass
#====================================================================
def default_consts():
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
#====================================================================
def set_consts( name_l, val_l, eos_d ):
    if 'const_d' in eos_d.keys():
        const_d = eos_d['const_d']
    else:
        init_consts( eos_d )

    for name, val in zip( name_l, val_l ):
        const_d[name] = val

    pass
#====================================================================
def get_consts( name_l, eos_d ):
    """
    Retrieve list of desired consts stored in eos_d['const_d']
    """
    const_d = eos_d['const_d']
    const_l = []
    for name in name_l:
        const_l.append( const_d[name] )

    return tuple( const_l )
#====================================================================
def set_params( name_l, val_l, eos_d ):
    if 'param_d' in eos_d.keys():
        param_d = eos_d['param_d']
    else:
        param_d = {}
        eos_d['param_d'] = param_d

    for name, val in zip( name_l, val_l ):
        param_d[name] = val

    pass
#====================================================================
def get_params( name_l, eos_d ):
    """
    Retrieve list of desired params stored in eos_d['param_d']
    """
    param_d = eos_d['param_d']
    param_l = []
    for name in name_l:
        param_l.append( param_d[name] )

    return tuple( param_l )
#====================================================================
def swap_params( name_l, eos_d ):
    """
    Retrieve list of desired params stored in eos_d['param_d']
    """

    # Use shallow copy to avoid unneeded duplication
    eos_swap_d = copy.copy( eos_d )
    # Use deep copy on params to ensure swap without affecting original
    param_swap_d = copy.deepcopy(eos_d['param_d'])

    eos_swap_d['param_d'] = param_swap_d

    set_params( name_l, eos_swap_d )

    return eos_swap_d
#====================================================================
def set_array_params( basename, param_arr_a, eos_d ):
    name_l = []

    for i in range(len(param_arr_a)):
        iname = basename+'_'+np.str(i)
        name_l.append(iname)

    set_params(name_l, param_arr_a, eos_d)
#====================================================================
def get_array_params( basename, eos_d ):
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
#====================================================================
def set_modtypes( name_l, val_l, eos_d ):
    if 'modtype_d' in eos_d.keys():
        modtype_d = eos_d['modtype_d']
    else:
        modtype_d = {}
        eos_d['modtype_d'] = modtype_d

    # Should we verify match?
    for name, val in zip( name_l, val_l ):
        modtype_d[name] = val
        # No longer functional test
        # if globals().has_key(name):
        #     # modtype = globals()[name]
        #     # modtype_d[name] = modtype()
        #     modtype_d[name] = val
        # else:
        #     print name + " is not a valid modtype object"

    pass
#====================================================================
def get_modtypes( name_l, eos_d ):
    """
    Retrieve list of desired model types stored in eos_d['modtype_d']
    """
    modtype_d = eos_d['modtype_d']
    modtype_l = []
    for name in name_l:
        modtype_l.append( modtype_d[name] )

    return tuple( modtype_l )
#====================================================================
def set_args( name_l, val_l, eos_d ):
    if 'arg_d' in eos_d.keys():
        arg_d = eos_d['arg_d']
    else:
        arg_d = {}
        eos_d['arg_d'] = arg_d

    for name, val in zip( name_l, val_l ):
        arg_d[name] = val

    pass
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
# ModFit Class (Is this even in use?)
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
