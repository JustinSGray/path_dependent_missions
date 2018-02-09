from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent


class FuelRateComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        num_points = self.metadata['num_nodes']

        self.add_input('thrust', shape=num_points)
        self.add_input('SFC_1em6_NNs', shape=num_points)
        self.add_output('m_dot', val=1.0, shape=num_points)

        arange = np.arange(num_points)
        self.declare_partials('m_dot', 'thrust', rows=arange, cols=arange)
        self.declare_partials('m_dot', 'SFC_1em6_NNs', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        thrust = inputs['thrust']
        SFC_NNs = inputs['SFC_1em6_NNs'] * 1e-6

        outputs['m_dot'] = -SFC_NNs * thrust / 9.80665

    def compute_partials(self, inputs, partials):
        thrust = inputs['thrust']
        SFC_NNs = inputs['SFC_1em6_NNs'] * 1e-6

        partials['m_dot', 'thrust'] = -SFC_NNs / 9.80665
        partials['m_dot', 'SFC_1em6_NNs'] = -thrust * 1e-6 / 9.80665
