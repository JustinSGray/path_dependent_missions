from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent


class SFCComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('propulsion_model')

    def setup(self):
        num_points = self.metadata['num_nodes']

        self.add_input('mach', shape=num_points)
        self.add_input('h', units='km', shape=num_points)
        self.add_input('throttle', shape=num_points)
        self.add_output('SFC_1em6_NNs', val=1.0, shape=num_points)

        self.x = np.zeros((num_points, 3))

        arange = np.arange(num_points)
        self.declare_partials('SFC_1em6_NNs', 'mach', rows=arange, cols=arange)
        self.declare_partials('SFC_1em6_NNs', 'h', rows=arange, cols=arange)
        self.declare_partials('SFC_1em6_NNs', 'throttle', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        M = inputs['mach']
        h_km = inputs['h']
        throttle = inputs['throttle']

        propulsion_model = self.metadata['propulsion_model']

        self.x[:, 0] = M
        self.x[:, 1] = h_km
        self.x[:, 2] = throttle

        outputs['SFC_1em6_NNs'] = propulsion_model.predict_values(self.x)[:, 1] * 2 * 1.e6

    def compute_partials(self, inputs, partials):
        M = inputs['mach']
        h_km = inputs['h']
        throttle = inputs['throttle']

        propulsion_model = self.metadata['propulsion_model']

        self.x[:, 0] = M
        self.x[:, 1] = h_km
        self.x[:, 2] = throttle

        partials['SFC_1em6_NNs', 'mach'] = \
            propulsion_model.predict_derivatives(self.x, 0)[:, 1] * 2 * 1.e6
        partials['SFC_1em6_NNs', 'h'] = \
            propulsion_model.predict_derivatives(self.x, 1)[:, 1] * 2 * 1.e6
        partials['SFC_1em6_NNs', 'throttle'] = \
            propulsion_model.predict_derivatives(self.x, 2)[:, 1] * 2 * 1.e6
