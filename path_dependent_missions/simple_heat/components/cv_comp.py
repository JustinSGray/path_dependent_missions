import numpy as np
from openmdao.api import ExplicitComponent


class CvComp(ExplicitComponent):
    """
    Compute the specific heat at constant volume, Cv, for JP-8 based on
    its temperature. Uses a linear curve fit from "Handbook of Aviation
    Fuel Properties," 1983.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        self.nn = self.options['num_nodes']

        self.add_input('T', shape=self.nn, units='K')
        self.add_output('Cv', shape=self.nn, units='J/(kg*K)')

        self.ar = ar = np.arange(self.nn)
        self.declare_partials('Cv', 'T', rows=ar, cols=ar)

        # self.declare_partials('*', '*', method='cs')
        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        T = inputs['T']

        # Simple linear fit from graph in book
        slope = (2.65 - 2.05) / (180 - 43)
        Cv = slope * (T - 316) + 2.05

        # Multiply by 1e3 to go from kJ to J in the numerator
        outputs['Cv'] = Cv * 1e3

    def compute_partials(self, inputs, partials):
        T = inputs['T']
        slope = (2.65 - 2.05) / (180 - 43)
        partials['Cv', 'T'] = slope * 1e3
