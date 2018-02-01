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
