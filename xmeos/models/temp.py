

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

