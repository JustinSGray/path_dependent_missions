import numpy as np
from openmdao.api import ExplicitComponent


class TankComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        self.nn = self.metadata['num_nodes']

        self.add_input('m', shape=self.nn, units='kg', val=1.)
        self.add_input('m_flow', shape=self.nn, units='kg/s')
        self.add_input('m_in', shape=self.nn, units='kg/s')
        self.add_input('T', shape=self.nn, units='K')
        self.add_input('T_in', shape=self.nn, units='K')
        self.add_output('m_dot', shape=self.nn, units='kg/s')
        self.add_output('m_out', shape=self.nn, units='kg/s')
        self.add_output('T_dot', shape=self.nn, units='K/s')
        self.add_output('T_out', shape=self.nn, units='K')

        self.Cv = 1.
        self.q_env = 10.

        self.ar = ar = np.arange(self.nn)
        self.declare_partials('T_out', 'T', val=1., rows=ar, cols=ar)
        self.declare_partials('m_out', 'm_flow', val=1., rows=ar, cols=ar)
        self.declare_partials('m_dot', 'm_in', val=1., rows=ar, cols=ar)
        self.declare_partials('m_dot', 'm_flow', val=-1., rows=ar, cols=ar)

        self.declare_partials('T_dot', 'T_in', rows=ar, cols=ar)
        self.declare_partials('T_dot', 'm_in', rows=ar, cols=ar)
        self.declare_partials('T_dot', 'm', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['m_dot'] = inputs['m_in'] - inputs['m_flow']
        outputs['T_dot'] = self.q_env / (inputs['m'] * self.Cv) + inputs['T_in'] * inputs['m_in'] / inputs['m']
        outputs['T_out'] = inputs['T']
        outputs['m_out'] = inputs['m_flow']

    def compute_partials(self, inputs, partials):
        partials['T_dot', 'T_in'] = inputs['m_in'] / inputs['m']
        partials['T_dot', 'm_in'] = inputs['T_in'] / inputs['m']
        partials['T_dot', 'm'] = -inputs['T_in'] * inputs['m_in'] / inputs['m']**2 - self.q_env / (inputs['m']**2 * self.Cv)
