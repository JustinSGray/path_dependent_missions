from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent


class SMTMaxThrustComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('propulsion_model')

    def setup(self):
        num_points = self.metadata['num_nodes']

        self.add_input('mach', shape=num_points)
        self.add_input('h', shape=num_points, units='km')
        self.add_output('max_thrust', shape=num_points, units='N')

        self.x = np.zeros((num_points, 3))
        self.x[:, 2] = 1.0

        arange = np.arange(num_points)
        self.declare_partials('max_thrust', 'mach', rows=arange, cols=arange)
        self.declare_partials('max_thrust', 'h', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        M = inputs['mach']
        h_km = inputs['h']

        propulsion_model = self.metadata['propulsion_model']

        self.x[:, 0] = M
        self.x[:, 1] = h_km

        outputs['max_thrust'] = propulsion_model.predict_values(self.x)[:, 0] * 2

    def compute_partials(self, inputs, partials):
        M = inputs['mach']
        h_km = inputs['h']

        propulsion_model = self.metadata['propulsion_model']

        self.x[:, 0] = M
        self.x[:, 1] = h_km

        partials['max_thrust', 'mach'] = \
            propulsion_model.predict_derivatives(self.x, 0)[:, 0] * 2
        partials['max_thrust', 'h'] = \
            propulsion_model.predict_derivatives(self.x, 1)[:, 0] * 2
