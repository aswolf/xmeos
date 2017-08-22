# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from future.utils import with_metaclass

import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod


__all__ = ['Eos','Calculator',
           'set_array_params', 'get_array_params', 'fill_array']





# xmeos.models.Calculator
# xmeos.models.Eos
#====================================================================
# EOS Model Classes
#====================================================================
class Eos(with_metaclass(ABCMeta)):
    """
    Abstract Equation of State Parent Base class
    """

    def __init__(self, natom=1):
        self._pre_init(natom=natom)

        ##########################
        # Model-specific initialization
        ##########################

        self._post_init()
        pass

    def _pre_init(self, natom=1, molar_mass=100):
        # self._init_all_calculators()
        self._calculators = {}
        self._natom=natom
        self._molar_mass=molar_mass
        pass

    def _post_init(self, model_state={}):

        param_names, param_units, param_defaults, param_scales = \
            self._get_calculator_params()

        param_values = self._overwrite_param_values(param_names, param_defaults,
                                                    model_state)

        self._param_names = param_names
        self._param_units = param_units
        self._param_scales = param_scales
        self._param_values = param_values

        pass

    def _overwrite_param_values(self, param_names, param_values, model_state):
        if not model_state:
            return param_values

        if ('param_names' not in model_state) or (
            set(model_state['param_names']) != set(param_names)):

            raise LookupError('model_state must contain param_names matching '
                              'calculator params: '+param_names)


        if ('param_values' not in model_state) or (
           len(model_state['param_values']) != len(model_state['param_names'])):
            raise LookupError('model_state must provide param_values '
                              'for each parameter.')

        state_names = np.array(model_state['param_names'])
        state_values = np.array(model_state['param_values'])

        for ind, param_name in enumerate(param_names):
            param_values[ind] = state_values[state_names==param_name]

        return param_values

    ######################
    # Calculator methods #
    ######################
    def _get_calculator_params( self ):
        """
        Get list of valid Eos calculators
        """

        # Initialize with ref parameters
        param_names = self._param_ref_names
        param_units = self._param_ref_units
        param_defaults = self._param_ref_defaults
        param_scales = self._param_ref_scales

        # Add all calculator parameters
        for calc in self._calculators:
            param_names.extend(self._calculators[calc].param_names)
            param_units.extend(self._calculators[calc].param_units)
            param_defaults.extend(self._calculators[calc].param_defaults)
            param_scales.extend(self._calculators[calc].param_scales)

        u, indices = np.unique(np.array(param_names), return_index=True)
        indices = np.sort(indices)

        param_names = np.array([param_names[ind] for ind in indices])
        param_units = np.array([param_units[ind] for ind in indices])
        param_defaults = np.array([param_defaults[ind] for ind in indices])
        param_scales = np.array([param_scales[ind] for ind in indices])

        return param_names, param_units, param_defaults, param_scales

    @property
    def calculators(self):
        return self._calculators

    def _add_calculator(self, calc, calc_type='compress'):
        """
        Store calculator instance in its correct place.

        Parameters
        ----------
        calc : Calculator obj
            Instance of Calculator().
        calc_type : {'compress','thermal','gamma','heat_capacity','thermal_exp'}
            Type of calculator object.

        """
        assert isinstance(calc,Calculator), \
            'calc must be a valid Calculator object instance.'

        self._calculators[calc_type] = calc
        pass

    #####################
    # Parameter methods #
    #####################
    def _get_required_param_names( self ):
        """
        Get param names from calculators.

        """
        param_names = {}

        # for calc in self._get_calculators():
        #     param_names.update(calc.param_names())

        for calc in self.calculators:
            param_names.update(self.calculators[calc].param_names())

        return param_names

    def _validate_param_names(self, param_names):
        """
        Check that all param names are valid.

        If param_names is None, replace with full parameter name list.

        """
        if param_names is None:
            param_names = self.param_names
        else:
            if isinstance(param_names, str):
                param_names = [param_names]

            assert all( name in self.param_names for name in param_names ),\
                'All provided param_names must be valid parameter names.'

        return param_names

    #####################
    #  Need to store param names, units, scales, values together
    #    - can be done with param_names=[], param_units={},
    #                       param_scales={}, param_values={}
    @property
    def natom(self):
        """
        Number of atoms (per working formula unit).

        """
        return self._natom

    @property
    def molar_mass(self):
        """
        Molar mass in g/mol of working formula unit.

        """
        return self._molar_mass

    def _get_element_index(self, param_names):
        sorter = np.argsort(self._param_names)
        ind_params = sorter[np.searchsorted(self._param_names, param_names,
                                            sorter=sorter)]
        return ind_params

    def _override_params(self, values, overrides):
        if overrides is not None:
            if len(overrides)!=len(values):
                raise LookupError('Overrides (if provided) must be a list '
                                  'of values with length equal to param_names. '
                                  'If no overrides are needed, use'
                                  'default value of None. If some overrides '
                                  'are needed, set elements equal to override '
                                  'value as needed and set rest to None.')

            for ind, override in enumerate(overrides):
                if override is not None:
                    values[ind] = override

        return values

    @property
    def param_names(self):
        """
        List of parameter names.

        Returns
        -------
        names : str list
            list of parameter names

        """
        return self._param_names

    def get_param_units(self, param_names=None):
        """
        Units for (selected) parameters.

        Parameters
        ----------
        param_names : list-like, default None
            if given, lists desired subset of parameters by name

        Returns
        -------
        units : array-like
            list of units for selected parameters

        """

        param_names = self._validate_param_names(param_names)
        ind_params = self._get_element_index(param_names)

        units = self._param_units[ind_params]

        return units

    @property
    def param_units(self):
        """
        Units for (selected) parameters.

        Parameters
        ----------
        param_names : list-like, default None
            if given, lists desired subset of parameters by name

        Returns
        -------
        units : array-like
            list of units for selected parameters

        """

        return self.get_param_units()

    def get_param_scales(self, param_names=None):
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

        param_names = self._validate_param_names(param_names)
        ind_params = self._get_element_index(param_names)

        scales = self._param_scales[ind_params]

        return scales

    def get_param_values(self, param_names=None, overrides=None):
        """
        Values for (selected) parameters.

        Parameters
        ----------
        param_names : str array
            list of parameter names

        Returns
        -------
        values : double array
            values of (selected) parameters

        """

        param_names = self._validate_param_names(param_names)
        ind_params = self._get_element_index(param_names)
        values = self._param_values[ind_params]
        values = self._override_params(values, overrides)

        return values

    def get_array_param_names(self, basename):
        """
        Get all param names for array parameter.

        Parameters
        ----------
        basename : str
            array parameter basename

        Returns
        -------
        param_names : str array
            list of parameter names

        """

        all_param_names = self._param_names
        param_names = []
        for name in all_param_names:
            if name.startswith('_'+basename+'_'):
                param_names.append(name)

        return param_names

    # def get_array_param_names(self, array_param_basename):
    #     Nchar = len(array_param_basename)
    #     param_names = []

    #     for key in self.param_names:
    #         if key.startswith(array_param_basename):

    #     return param_names

    # def get_array_param_values(self, array_param_basename):
    #     """
    #     Array values for array-type parameter.

    #     Parameters
    #     ----------
    #     array_param_basename : str
    #         basename of array parameter

    #     Returns
    #     -------
    #     values : double array
    #         values of array parameter

    #     """

    #     param_names = self._validate_param_names(param_names)
    #     ind_params = self._get_element_index(param_names)
    #     values = self._param_values[ind_params]
    #     values = self._override_params(values, overrides)

    #     return values

    def set_param_values(self, param_values, param_names=None):
        """
        Set values for (selected) parameters.

        Parameters
        ----------
        param_values : double array
            list of parameter values
        param_names : list, default None
            if given, lists desired subset of parameters by name

        """

        param_names = self._validate_param_names(param_names)
        try:
            len(param_values)
        except:
            param_values = [param_values]

        assert len(param_names)==len(param_values), \
            'param_names and param_values must have the same length'

        ind_params = self._get_element_index(param_names)
        self._param_values[ind_params] = param_values
        # for ind, value in enumerate(param_values):
        #     self._param_values[ind] = value

        pass

    @property
    def param_values(self):
        """
        Values for (selected) parameters.

        Parameters
        ----------
        param_names : str array
            list of parameter names

        Returns
        -------
        values : double array
            values of (selected) parameters

        """
        return self.get_param_values()

    @param_values.setter
    def param_values(self, param_values):
        """
        Set values for (selected) parameters.

        Parameters
        ----------
        param_values : double array
            list of parameter values
        param_names : list, default None
            if given, lists desired subset of parameters by name

        """

        self.set_param_values(param_values)
        pass

    @property
    def model_state(self):
        model_state = {
            'param_names': self.param_names.tolist(),
            'param_values':self.param_values.tolist(),
        }
        return model_state
#====================================================================
class Calculator(with_metaclass(ABCMeta)):
    """
    Abstract Equation of State Calculator

    """

    def __init__(self, eos_mod):
        self._eos_mod = eos_mod
        self._init_params()
        self._init_required_calculators()

        pass

    def _init_required_calculators(self):
        """Initialize list of other required calculators."""
        self._required_calculators = None
        pass

    @property
    def name(self):
        objname = self.__class__.__name__
        # Remove leading underscore
        return objname[1:]

    @property
    def eos_mod(self):
        """
        Parent Eos Model.

        """
        return self._eos_mod

    @property
    def param_names(self):
        """
        List of parameter names for this Eos Calculator.

        """
        return self._param_names

    @property
    def param_units(self):
        """
        List of parameter units for this Eos Calculator.

        """
        return self._param_units

    @property
    def param_defaults(self):
        """
        List of parameter default values for this Eos Calculator.

        """
        return self._param_defaults

    def _validate_param_names(self, param_names):
        """
        Check that all param names are valid.

        If param_names is None, replace with full parameter name list.

        """
        if param_names is None:
            param_names = self.param_names
        else:
            if isinstance(param_names, str):
                param_names = [param_names]

            assert all( name in self.param_names for name in param_names ),\
                'All provided param_names must be valid parameter names.'

        return param_names

    def _set_params(self, param_names, param_units,
                    param_defaults, param_scales, order=None):
        self._param_names = np.array(param_names)
        self._param_units = np.array(param_units)
        self._param_defaults = np.array(param_defaults)
        self._param_scales = np.array(param_scales)
        self._order = order
        pass

    def get_param_defaults(self, param_names=None):
        """
        Values for (selected) parameters.

        Parameters
        ----------
        param_names : str array
            list of parameter names

        Returns
        -------
        values : double array
            values of (selected) parameters

        """
        param_names = self._validate_param_names(param_names)
        # NOTE need to return defaults in order provided!!!
        defaults = []
        for name in param_names:
            ind, = np.where(self._param_names==name)
            defaults.append(self._param_defaults[ind])

        defaults = np.array(defaults)
        return defaults

    @property
    def param_scales(self):
        """
        List of parameter scale values for this Eos Calculator.

        """
        return self._param_scales

    @property
    def required_calculators(self):
        """
        List of required coupled calculator types.

        This Eos Calculator requires other calculators to function. This
        provides a list of those calculator types.

        """
        return self._required_calculators
#====================================================================

#====================================================================
CONSTS = {}
CONSTS['eVperHa'] = 27.211 # eV/Ha
CONSTS['JperHa'] = 4.35974434e-18 # J/Ha
CONSTS['JperCal'] = 4.184 # J/Cal
CONSTS['Nmol'] = 6.0221413e+23 # atoms/mol
CONSTS['kJ_molpereV'] = 96.49 # kJ/mol/eV
CONSTS['R'] = 8.314462 # J/K/mol
CONSTS['kboltz'] = 8.617332e-5 # eV/K
CONSTS['ang3percc'] = 1e24 # ang^3/cm^3
CONSTS['PV_ratio'] = 160.2176487 # (GPa*ang^3)/eV
CONSTS['TS_ratio'] = CONSTS['R']/CONSTS['kboltz'] # (J/mol)/eV

#====================================================================
def simplify_poly(shifted_coefs):
    """
    convert a polynomial from shifted to absolute form

    sum_i ( b_i*(x-1)**i )  =>  sum_k ( c_k * x**k )
    """

    rev_shifted_coefs = np.flipud(shifted_coefs)

    rev_abs_coefs = np.zeros(shifted_coefs.shape)
    for i,icoef in enumerate(rev_shifted_coefs):
        i_poly_expand = icoef*sp.special.binom(i,range(i+1))*(-1)**np.arange(i+1)
        rev_abs_coefs[0:i+1] += np.flipud(i_poly_expand)

    abs_coefs= np.flipud(rev_abs_coefs)
    return abs_coefs
#====================================================================
def make_array_param_names(basename, order, skipzero=False):
    if skipzero:
        first = 1
    else:
        first = 0

    exp_ind = np.arange(first, order)
    coef_index = exp_ind.astype(str)
    param_names = ['_' + basename + '_' + index_str for index_str in coef_index]

    return param_names
#====================================================================
def get_array_param_index(param_names):
    coef_index = [int(np.str.split(name,'_')[-1]) for name in param_names]
    return coef_index
#====================================================================
def make_array_param_defaults(param_names, base_scale=500, deriv_scale=3,
                              base_unit='GPa', deriv_unit='ang', deriv_exp=3):

    coef_index =get_array_param_index(param_names)
    order = np.max(coef_index)

    param_units = []
    param_scales = []
    param_defaults = []

    for ind in coef_index:
        if ind==0:
            param_units.append(base_unit)
            param_scales.append(base_scale)
            param_defaults.append(base_scale)
        else:
            expstr = str(ind*deriv_exp)
            scale = base_scale/deriv_scale**ind
            default = scale*(-1)**ind
            param_units.append(base_unit+'/'+deriv_unit+'^'+expstr)
            param_scales.append(scale)
            param_defaults.append(default)

    return param_units, param_defaults, param_scales
#====================================================================
def make_array_param_units(param_names, base_unit='GPa', deriv_unit='(ang^3)'):

    coef_index =get_array_param_index(param_names)
    order = np.max(coef_index)

    param_units = []

    for ind in coef_index:
        if ind==0:
            param_units.append(base_unit)
        else:
            param_units.append(base_unit+'/'+deriv_unit+'^'+str(ind))

    return param_units
#====================================================================
def get_consts( keys ):
    # assert all((key in CONSTS for key in keys)), (
    #     'That is not a valid CONST name. Valid names are '+str(CONSTS.keys()) )

    return tuple(CONSTS[key] for key in keys)
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
def fill_array(*arg):
    """
    fix fill_array such that it returns two numpy arrays of equal size

    use numpy.full_like

    """
    Narg = len(arg)

    array_list = []
    for var in arg:
        var_a = np.asarray(var)
        if var_a.shape==():
            var_a = np.asarray([var_a])

        array_list.append(var_a)

    assert len(array_list)<=2, 'fill_array can only accept one or 2 arrays'

    if len(array_list)==1:
        return array_list[0]
    else:
        var1_a, var2_a = array_list[0], array_list[1]
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
