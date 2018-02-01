
class CompressMod(with_metaclass(ABCMeta, EosMod)):
    """
    Abstract Equation of State class for a reference Compression Path

    Path can either be isothermal (T=const) or adiabatic (S=const)

    For this restricted path, thermodyn properties depend only on volume

    """

    path_opts = ['T','S']

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
            scale_a, paramkey_a = self.get_param_scale_sub( eos_d )
            scale_a = np.append(scale_a,0.01)
            paramkey_a = np.append(paramkey_a,'logPmin')

            # paramkey_pos_a = np.append(paramkey_pos_a,1.0)

            # scale_neg_a, paramkey_neg_a = self.expand_adj_mod.get_param_scale_sub( eos_d )

            # ind_pos_a = self.validate_shared_param_scale(scale_pos_a,paramkey_pos_a,
            #                                              scale_neg_a,paramkey_neg_a)

            # # Since negative expansion EOS model params are a superset of those
            # # required for the positive compression model, we can simply return the
            # # scale and paramkey values from the negative expansion model
            # scale_a = scale_neg_a
            # paramkey_a = paramkey_neg_a

            # if output_ind:
            #     return scale_a, paramkey_a, ind_pos_a
            # else:
            #     return scale_a, paramkey_a
            return scale_a, paramkey_a

    def get_ind_expand( self, V_a, eos_d ):
        V0 = core.get_params( ['V0'], eos_d )
        ind_exp = np.where( V_a > V0 )[0]
        return ind_exp

    def get_path_const( self ):
        return self.path_const

    def get_level_const( self ):
        return self.level_const

    # EOS property functions
    def param_deriv( self, fname, paramname, V_a, eos_d, dxfrac=1e-6):
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
            param = core.get_params( [paramname], eos_d )[0]
            dparam = scale*dxfrac
            # print 'param: ' + np.str(param)
            # print 'dparam: ' + np.str(dparam)
        except:
            assert False, 'This is not a valid parameter name'

        # set param value in eos_d dict
        core.set_params( [paramname,], [param+dparam,], eos_d )

        # Note that self is implicitly included
        dval_a = fun(V_a, eos_d) - val0_a

        # reset param to original value
        core.set_params( [paramname], [param], eos_d )

        deriv_a = dval_a/dxfrac
        return deriv_a

    # NEED to write infer volume function

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
