import numpy as np
import eoslib

class TestCompressMod(object):
    param_name_l = ['V0', 'K0', 'KP0', 'E0']
    param_val_a = np.array([250.0, 150.0, 4.0, 0.0])
    # param_d = {'V0': 250.0, 'K0':150.0, 'KP0':4.0, 'E0':0.0}
    eos_d = {}

    def __init__(self):
        eoslib.init_const( self.eos_d )
        eoslib.set_modtype( [], [], self.eos_d )
        eoslib.set_param( param_name_l, param_val_a, self.eos_d)

class TestBirchMurn3(TestCompressMod):

    def __init__(self):
        TestCompressMod.__init__(self)
        eoslib.set_modtype(self.eos_d)

    def test_BirchMurn3(self):
        eos_d = self.eos_d

