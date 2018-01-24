from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp

from pointer.examples.min_time_climb.prop.bryson_thrust_comp import BrysonThrustComp

SHOW_PLOTS = True


class TestBrysonThrustComp(unittest.TestCase):

    @unittest.skipIf(not SHOW_PLOTS, 'this test is for visual confirmation, requires plotting')
    def test_other_values(self):
        n = 5

        p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='alt', val=np.zeros(n), units='ft')
        ivc.add_output(name='mach', val=np.zeros(n), units=None)

        p.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['alt', 'mach'])
        p.model.add_subsystem(name='tcomp', subsys=BrysonThrustComp(num_nodes=n))

        p.model.connect('alt', 'tcomp.alt')
        p.model.connect('mach', 'tcomp.mach')

        p.setup()

        # Values of alt and mach at our test points
        h = [0, 100, 1000, 2000.0, 65000.0]
        M = [0.29386358, 0.29386358, .1,  0.2, 1.0]

        p['alt'] = h
        p['mach'] = M

        p.run_model()

        thrust = p['tcomp.thrust']

        print(thrust)

        # THR = np.reshape(thrust, (5, 5))
        #
        # import matplotlib.pyplot as plt
        # plt.contourf(hh, MM, THR/np.max(THR), levels=np.linspace(0,1.0,100))
        # plt.colorbar()
        # plt.show()


if __name__ == '__main__':
    unittest.main()
