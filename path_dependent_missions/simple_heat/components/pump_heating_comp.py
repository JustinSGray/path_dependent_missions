import numpy as np
from openmdao.api import ExplicitComponent


class PumpHeatingComp(ExplicitComponent):
    """
    This component computes amount of heat added to the system based on the mass
    flow rate of the main fuel pump.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('heat_coeff', types=float)

    def setup(self):
        self.nn = self.options['num_nodes']
        self.heat_coeff = self.options['heat_coeff']

        self.add_input('m_flow', shape=self.nn, units='kg/s')
        self.add_output('Q_pump', shape=self.nn, units='W')

        self.ar = ar = np.arange(self.nn)
        self.declare_partials('Q_pump', 'm_flow', val=self.heat_coeff, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['Q_pump'] = self.heat_coeff * inputs['m_flow']
