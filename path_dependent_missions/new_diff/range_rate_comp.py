from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent

class RangeRateComp(ExplicitComponent):
    """
    The equation of motion governing range.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('TAS', val=np.ones(nn), desc='True airspeed', units='m/s')

        self.add_input('gam', val=np.zeros(nn), desc='Flight path angle', units='rad')

        self.add_output('r_dot', val=np.ones(nn), desc='Velocity along the ground (no wind)',
                        units='m/s')

        # Setup partials
        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(of='*', wrt='*', dependent=False)
        self.declare_partials(of='r_dot', wrt='TAS', rows=ar, cols=ar)
        self.declare_partials(of='r_dot', wrt='gam', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        TAS = inputs['TAS']
        gam = inputs['gam']
        outputs['r_dot'] = TAS*np.cos(gam)

    def compute_partials(self, inputs, partials):
        TAS = inputs['TAS']
        gam = inputs['gam']

        partials['r_dot', 'TAS'] = np.cos(gam)
        partials['r_dot', 'gam'] = -TAS * np.sin(gam)
