from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent

class PassThroughComp(ExplicitComponent):
    """
    The equation of motion governing range.
    """
    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_input('h_dot_in', val=np.ones(nn), units='m/s')
        self.add_input('r_dot_in', val=np.ones(nn), units='m/s')

        self.add_output('h_dot_out', val=np.ones(nn), units='m/s')
        self.add_output('r_dot_out', val=np.ones(nn), units='m/s')

        ar = np.arange(self.metadata['num_nodes'])
        self.declare_partials(of='*', wrt='*', dependent=False)
        self.declare_partials(of='h_dot_out', wrt='h_dot_in', rows=ar, cols=ar, val=1.)
        self.declare_partials(of='r_dot_out', wrt='r_dot_in', rows=ar, cols=ar, val=1.)

    def compute(self, inputs, outputs):
        outputs['h_dot_out'] = inputs['h_dot_in']
        outputs['r_dot_out'] = inputs['r_dot_in']
