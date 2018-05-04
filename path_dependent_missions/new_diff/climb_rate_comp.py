from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent

class ClimbRateComp(ExplicitComponent):
    """
    The equation of motion governing range.
    """
    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_input('TAS', val=np.ones(nn), desc='true airspeed', units='m/s')
        self.add_input('gam', val=np.ones(nn), desc='flight path angle', units='rad')

        self.add_output('h_dot', val=np.ones(nn), desc='rate of climb',
                        units='m/s')

        ar = np.arange(self.metadata['num_nodes'])
        self.declare_partials(of='*', wrt='*', dependent=False)
        self.declare_partials(of='h_dot', wrt='TAS', rows=ar, cols=ar)
        self.declare_partials(of='h_dot', wrt='gam', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['h_dot'] = inputs['TAS'] * np.sin(inputs['gam'])

    def compute_partials(self, inputs, partials):
        partials['h_dot', 'TAS'] = np.sin(inputs['gam'])
        partials['h_dot', 'gam'] = inputs['TAS']*np.cos(inputs['gam'])
