import numpy as np
from openmdao.api import ExplicitComponent


class TankComp(ExplicitComponent):
    """
    This contains information about the fuel tank, especially its temperature,
    mass of fuel, flow in and out, and how those all change.
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('q', types=float)

    def setup(self):
        self.nn = self.metadata['num_nodes']
        self.q = self.metadata['q']

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

        self.ar = ar = np.arange(self.nn)
        self.declare_partials('T_out', 'T', val=1., rows=ar, cols=ar)
        self.declare_partials('m_out', 'm_flow', val=1., rows=ar, cols=ar)
        self.declare_partials('m_dot', 'm_in', val=1., rows=ar, cols=ar)
        self.declare_partials('m_dot', 'm_flow', val=-1., rows=ar, cols=ar)

        self.declare_partials('T_dot', 'T_in', rows=ar, cols=ar)
        self.declare_partials('T_dot', 'T', rows=ar, cols=ar)
        self.declare_partials('T_dot', 'm_in', rows=ar, cols=ar)
        self.declare_partials('T_dot', 'm', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        # The change in fuel mass is the difference between the amount pumped out
        # and the amount coming back in from recirculation
        outputs['m_dot'] = inputs['m_in'] - inputs['m_flow']
        outputs['m_out'] = inputs['m_flow']

        # Both the heat energy being added to the fuel tank and the energy
        # coming from the recirculated fuel affect the temperature of the tank
        heat_input = self.q / (inputs['m'] * self.Cv)
        temp_input = (inputs['T_in'] - inputs['T']) * inputs['m_in'] / inputs['m']
        outputs['T_dot'] = heat_input + temp_input
        outputs['T_out'] = inputs['T']

    def compute_partials(self, inputs, partials):
        partials['T_dot', 'T_in'] = inputs['m_in'] / inputs['m']
        partials['T_dot', 'T'] = -inputs['m_in'] / inputs['m']
        partials['T_dot', 'm_in'] = (inputs['T_in'] - inputs['T']) / inputs['m']
        partials['T_dot', 'm'] = (inputs['T'] - inputs['T_in']) * inputs['m_in'] / inputs['m']**2 - self.q / (inputs['m']**2 * self.Cv)
