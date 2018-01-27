import numpy as np
from openmdao.api import ExplicitComponent

# TODO: see if this is actually needed. Currently it prevents numerical issues
# when m_in is 0.
tol = 1e-10

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

        self.Cv = 1.

        self.ar = ar = np.arange(self.nn)
        self.declare_partials('T_out', 'T_in', val=1., rows=ar, cols=ar)
        self.declare_partials('T_out', 'm_in', dependent=True, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['T_out'] = self.q / (inputs['m_in'] * self.Cv + tol) + inputs['T_in']
        print(self.name, inputs['T_in'], outputs['T_out'])

    def compute_partials(self, inputs, partials):
        partials['T_out', 'm_in'] = -self.Cv * self.q / (self.Cv * inputs['m_in'] + tol)**2
