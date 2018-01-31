import numpy as np
from openmdao.api import ExplicitComponent


class HeatExchangerComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('q', types=float)

    def setup(self):
        self.nn = self.metadata['num_nodes']
        self.q = self.metadata['q']

        self.add_input('T_in', shape=self.nn, units='K')
        self.add_input('m_in', shape=self.nn, units='kg/s')
        self.add_output('T_out', shape=self.nn, units='K')

        self.coeff = 10.

        self.ar = ar = np.arange(self.nn)
        self.declare_partials('T_out', 'T_in', val=1., rows=ar, cols=ar)
        self.declare_partials('T_out', 'm_in', val=self.coeff * self.q, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['T_out'] = self.coeff * self.q * inputs['m_in'] + inputs['T_in']
