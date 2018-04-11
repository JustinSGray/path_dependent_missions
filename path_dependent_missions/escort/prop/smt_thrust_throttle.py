from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent

from path_dependent_missions.F110.smt_model import get_F110_interp


scaler = 1.

class SMTThrustComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        num_points = self.metadata['num_nodes']
        self.prop_model = get_F110_interp()

        self.add_input('mach', shape=num_points)
        self.add_input('h', shape=num_points, units='ft')
        self.add_input('throttle', shape=num_points)
        self.add_output('thrust', shape=num_points, units='lbf')
        self.add_output('m_dot', shape=num_points, units='lbm/s')

        self.x = np.zeros((num_points, 3))

        arange = np.arange(num_points)
        self.declare_partials('thrust', 'mach', rows=arange, cols=arange)
        self.declare_partials('thrust', 'h', rows=arange, cols=arange)
        self.declare_partials('thrust', 'throttle', rows=arange, cols=arange)

        self.declare_partials('m_dot', 'mach', rows=arange, cols=arange)
        self.declare_partials('m_dot', 'h', rows=arange, cols=arange)
        self.declare_partials('m_dot', 'throttle', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        self.x[:, 0] = inputs['mach']
        self.x[:, 1] = inputs['h'] / 1e4
        self.x[:, 2] = inputs['throttle']

        smt_out = self.prop_model.predict_values(self.x)

        outputs['thrust'] = smt_out[:, 0] * 2 * 1e4 / scaler
        outputs['m_dot'] = -smt_out[:, 1] * 2

    def compute_partials(self, inputs, partials):
        self.x[:, 0] = inputs['mach']
        self.x[:, 1] = inputs['h'] / 1e4
        self.x[:, 2] = inputs['throttle']

        mach_derivs = self.prop_model.predict_derivatives(self.x, 0)
        h_derivs = self.prop_model.predict_derivatives(self.x, 1)
        throttle_derivs = self.prop_model.predict_derivatives(self.x, 2)

        partials['thrust', 'mach'] = mach_derivs[:, 0] * 2 * 1e4 / scaler
        partials['thrust', 'h'] = h_derivs[:, 0] * 2 * 1e4 / 1e4 / scaler
        partials['thrust', 'throttle'] = throttle_derivs[:, 0] * 2 * 1e4 / scaler

        partials['m_dot', 'mach'] = -mach_derivs[:, 1] * 2
        partials['m_dot', 'h'] = -h_derivs[:, 1] * 2 / 1e4
        partials['m_dot', 'throttle'] = -throttle_derivs[:, 1] * 2


if __name__ == "__main__":
    from openmdao.api import Problem, view_model, IndepVarComp, Group

    nn = 3

    p = Problem(model=Group())

    p.model.add_subsystem('smt', SMTThrustComp(num_nodes=nn), promotes=['*'])

    p.setup(check=True)
    p.run_model()
    p.check_partials(compact_print=True)

    # view_model(p)
