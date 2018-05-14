from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent

from esav.run.smt_model import get_ESAV_interp, get_data


scaler = 2.
drag_scaler = 1.
lift_scaler = 10.

class AeroSMTComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_points = self.options['num_nodes']

        self.aero_model = get_ESAV_interp()

        self.add_input('mach', shape=num_points)
        self.add_input('h', shape=num_points, units='km')
        self.add_input('alpha', shape=num_points, units='deg')
        self.add_output('CL', shape=num_points)
        self.add_output('CD', shape=num_points)

        self.x = np.zeros((num_points, 3))

        arange = np.arange(num_points)
        self.declare_partials('CL', 'mach', rows=arange, cols=arange)
        self.declare_partials('CL', 'h', rows=arange, cols=arange)
        self.declare_partials('CL', 'alpha', rows=arange, cols=arange)
        self.declare_partials('CD', 'mach', rows=arange, cols=arange)
        self.declare_partials('CD', 'h', rows=arange, cols=arange)
        self.declare_partials('CD', 'alpha', rows=arange, cols=arange)

        self.set_check_partial_options('*', method='fd')

    def compute(self, inputs, outputs):
        aero_model = self.aero_model

        self.x[:, 0] = inputs['mach']
        self.x[:, 1] = inputs['h'] / 1e4
        self.x[:, 2] = inputs['alpha'] / 10.

        outputs['CL'] = aero_model.predict_values(self.x)[:, 0] * scaler * lift_scaler
        outputs['CD'] = aero_model.predict_values(self.x)[:, 1] * scaler * drag_scaler

    def compute_partials(self, inputs, partials):
        aero_model = self.aero_model

        self.x[:, 0] = inputs['mach']
        self.x[:, 1] = inputs['h'] / 1e4
        self.x[:, 2] = inputs['alpha'] / 10.

        derivs_mach = aero_model.predict_derivatives(self.x, 0) * scaler
        derivs_h = aero_model.predict_derivatives(self.x, 1) * scaler
        derivs_alpha = aero_model.predict_derivatives(self.x, 2) * scaler

        partials['CL', 'mach'] = derivs_mach[:, 0] * lift_scaler
        partials['CL', 'h'] = derivs_h[:, 0] / 1e4 * lift_scaler
        partials['CL', 'alpha'] = derivs_alpha[:, 0] / 10 * lift_scaler

        partials['CD', 'mach'] = derivs_mach[:, 1] * drag_scaler
        partials['CD', 'h'] = derivs_h[:, 1] / 1e4 * drag_scaler
        partials['CD', 'alpha'] = derivs_alpha[:, 1] / 10 * drag_scaler
