from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent

class TASRateComp(ExplicitComponent):
    """
    Compute TAS rate, given range rate and h rate
    """
    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_input('r_dot', val=np.ones(nn), desc='rate of change of range', units='m/s')
        self.add_input('h_dot', val=np.ones(nn), desc='rate of change of altitude', units='m/s')
        self.add_input('r_dot_rate', val=np.ones(nn), desc='rate of change of rate of change of range', units='m/s**2')
        self.add_input('h_dot_rate', val=np.ones(nn), desc='rate of change of rate of change of altitude', units='m/s**2')

        self.add_output('gam', val=np.ones(nn), desc='flight path angle', units='rad')
        self.add_output('TAS_rate', val=np.ones(nn), desc='rate of change of TAS', units='m/s**2')
        self.add_output('gam_dot', val=np.ones(nn), desc='rate of change of flight path angle', units='rad/s')

        # Setup partials
        # ar = np.arange(self.metadata['num_nodes'])
        # self.declare_partials(of='TAS_rate', wrt=['r_dot', 'h_dot', 'h_dot_rate', 'r_dot_rate'], rows=ar, cols=ar)
        # self.declare_partials(of='gam', wrt=['h_dot', 'r_dot'], rows=ar, cols=ar)
        # self.declare_partials(of='gam_dot', wrt=['h_dot_rate', 'r_dot_rate'], rows=ar, cols=ar)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        dh = inputs['h_dot']
        dr = inputs['r_dot']
        ddh = inputs['h_dot_rate']
        ddr = inputs['r_dot_rate']

        # TODO : check these computes and derivs
        outputs['TAS_rate'] = (dh * ddh - dr * ddr) / np.sqrt(dh**2 + dr**2)
        outputs['gam'] = np.arctan2(dh, dr)
        outputs['gam_dot'] = (dr * ddh - dh * ddr) / (ddh + ddr)

    # def compute_partials(self, inputs, partials):
    #     dh = inputs['h_dot']
    #     dr = inputs['r_dot']
    #     ddh = inputs['h_dot_rate']
    #     ddr = inputs['r_dot_rate']
    #
    #     TAS_rate_denom = 1/np.sqrt(dr**2 + dh**2)
    #
    #     partials['TAS_rate', 'h_dot'] = dr * (ddh * dr - dh * ddr) / (dh**2 + dr**2)**1.5
    #     partials['TAS_rate', 'h_dot_rate'] = dh / np.sqrt(dh**2 + dr**2)
    #     partials['TAS_rate', 'r_dot'] = dh * (dh * ddr - ddh * dr) / (dh**2 + dr**2)**1.5
    #     partials['TAS_rate', 'r_dot_rate'] = dr / np.sqrt(dh**2 + dr**2)
    #
    #     partials['gam', 'r_dot'] = -dh * TAS_rate_denom**2
    #     partials['gam', 'h_dot'] = dr * TAS_rate_denom**2
    #
    #     partials['gam_dot', 'r_dot_rate'] = 1.
    #     partials['gam_dot', 'h_dot_rate'] = 1.

if __name__ == "__main__":
    from openmdao.api import Problem, view_model, IndepVarComp, Group

    nn = 3

    p = Problem(model=Group())

    ivc = IndepVarComp()
    ivc.add_output('h_dot', shape=nn, val=1.2)
    ivc.add_output('r_dot', shape=nn, val=1.3332)
    ivc.add_output('h_dot_rate', shape=nn, val=.22)
    ivc.add_output('r_dot_rate', shape=nn, val=.882)

    p.model.add_subsystem('ivc', ivc, promotes=['*'])

    p.model.add_subsystem('ras', TASRateComp(num_nodes=nn), promotes=['*'])

    p.setup(check=True)
    p.run_model()
    p.check_partials(compact_print=True)

    # view_model(p)
