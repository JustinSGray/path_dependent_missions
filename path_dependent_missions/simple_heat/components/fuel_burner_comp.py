import numpy as np
from openmdao.api import ExplicitComponent


class FuelBurnerComp(ExplicitComponent):
    """
    This component computes the amount of fuel to be recirculated based on the
    total fuel coming in and the amount of fuel being burned by the engine.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        self.nn = self.options['num_nodes']

        self.add_input('m_in', shape=self.nn, units='kg/s')
        self.add_input('m_burn', shape=self.nn, units='kg/s')
        self.add_output('m_recirculated', shape=self.nn, units='kg/s')

        self.ar = ar = np.arange(self.nn)
        self.declare_partials('m_recirculated', 'm_in', val=1., rows=ar, cols=ar)
        self.declare_partials('m_recirculated', 'm_burn', val=-1., rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['m_recirculated'] = inputs['m_in'] - inputs['m_burn']
