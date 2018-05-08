import numpy as np
from openmdao.api import ExplicitComponent


class EngineHeatingComp(ExplicitComponent):
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

        self.add_input('throttle', shape=self.nn, units=None)
        self.add_output('Q_engine', shape=self.nn, units='W')

        self.ar = ar = np.arange(self.nn)
        self.declare_partials('Q_engine', 'throttle', val=self.heat_coeff, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['Q_engine'] = self.heat_coeff * inputs['throttle']
