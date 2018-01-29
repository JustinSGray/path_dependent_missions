import numpy as np
from openmdao.api import ExplicitComponent


class PowerComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        self.nn = self.metadata['num_nodes']

        self.add_input('m_flow', shape=self.nn, units='kg/s')
        self.add_output('power', shape=self.nn, units='W')

        self.ar = ar = np.arange(self.nn)
        self.declare_partials('power', 'm_flow', val=1., rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['power'] = inputs['m_flow']
