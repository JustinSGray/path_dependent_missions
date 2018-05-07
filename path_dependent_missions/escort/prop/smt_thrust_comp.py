from __future__ import division
import numpy as np

from openmdao.api import ExplicitComponent

from path_dependent_missions.F110.smt_model import get_F110_interp


class SMTMaxThrustComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('propulsion_model')

    def setup(self):
        num_points = self.options['num_nodes']
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
        self.x[:, 0] = inputs['mach']
        self.x[:, 1] = inputs['h'] / 1e4

        smt_out = self.prop_model.predict_values(self.x)

        outputs['max_thrust'] = smt_out[:, 0] * 2 * 1e4

    def compute_partials(self, inputs, partials):
        self.x[:, 0] = inputs['mach']
        self.x[:, 1] = inputs['h'] / 1e4

        mach_derivs = self.prop_model.predict_derivatives(self.x, 0)
        h_derivs = self.prop_model.predict_derivatives(self.x, 1)

        partials['max_thrust', 'mach'] = mach_derivs[:, 0] * 2 * 1e4
        partials['max_thrust', 'h'] = h_derivs[:, 0] * 2 * 1e4 / 1e4

if __name__ == "__main__":
    from openmdao.api import Problem, view_model, IndepVarComp, Group

    nn = 3

    p = Problem(model=Group())

    p.model.add_subsystem('smt', SMTMaxThrustComp(num_nodes=nn), promotes=['*'])

    p.setup(check=True)
    p.run_model()
    p.check_partials(compact_print=True)

    # view_model(p)
