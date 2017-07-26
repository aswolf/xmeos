

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
