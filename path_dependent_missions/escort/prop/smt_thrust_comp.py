from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent

from path_dependent_missions.F110.smt_model import get_F110_interp


class SMTMaxThrustComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('propulsion_model')

    def setup(self):
        num_points = self.metadata['num_nodes']
        self.prop_model = get_F110_interp()

        self.add_input('mach', shape=num_points)
        self.add_input('h', shape=num_points, units='ft')
        self.add_output('max_thrust', shape=num_points, units='lbf')

        self.x = np.zeros((num_points, 3))
        self.x[:, 2] = 1.0

        arange = np.arange(num_points)
        self.declare_partials('max_thrust', 'mach', rows=arange, cols=arange)
        self.declare_partials('max_thrust', 'h', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        M = inputs['mach']
        h = inputs['h'] / 1e4

        self.x[:, 0] = M
        self.x[:, 1] = h

        outputs['max_thrust'] = self.prop_model.predict_values(self.x)[:, 0] * 2 * 1e4

    def compute_partials(self, inputs, partials):
        M = inputs['mach']
        h = inputs['h']

        self.x[:, 0] = M
        self.x[:, 1] = h / 1e4

        partials['max_thrust', 'mach'] = \
            self.prop_model.predict_derivatives(self.x, 0)[:, 0] * 2 * 1e4 / 1e4
        partials['max_thrust', 'h'] = \
            self.prop_model.predict_derivatives(self.x, 1)[:, 0] * 2 * 1e4 / 1e4
