import numpy as np
from openmdao.api import ExplicitComponent


class TankAloneComp(ExplicitComponent):
    """
    This contains information about the fuel tank, especially its temperature,
    mass of fuel, flow in and out, and how those all change.

    This is using the simplified equation from Alyanak's paper.
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        self.nn = self.metadata['num_nodes']

        self.add_input('m', shape=self.nn, units='kg', val=1.)
        self.add_input('T', shape=self.nn, units='K')
        self.add_input('m_flow', shape=self.nn, units='kg/s')
        self.add_input('m_burn', shape=self.nn, units='kg/s')
        self.add_input('Q_env', shape=self.nn, units='W')
        self.add_input('Q_sink', shape=self.nn, units='W')
        self.add_input('Q_out', shape=self.nn, units='W')

        self.add_output('m_dot', shape=self.nn, units='kg/s')
        self.add_output('T_dot', shape=self.nn, units='K/s')
        self.add_output('m_constraint', shape=self.nn, units='kg/s')

        self.Cv = 1.

        # self.ar = ar = np.arange(self.nn)
        # self.declare_partials('T_out', 'T', val=1., rows=ar, cols=ar)
        # self.declare_partials('m_out', 'm_flow', val=1., rows=ar, cols=ar)
        # self.declare_partials('m_dot', 'm_in', val=1., rows=ar, cols=ar)
        # self.declare_partials('m_dot', 'm_flow', val=-1., rows=ar, cols=ar)
        #
        # self.declare_partials('T_dot', 'T_in', rows=ar, cols=ar)
        # self.declare_partials('T_dot', 'T', rows=ar, cols=ar)
        # self.declare_partials('T_dot', 'm_in', rows=ar, cols=ar)
        # self.declare_partials('T_dot', 'm', rows=ar, cols=ar)

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        Q_env = inputs['Q_env']
        Q_sink = inputs['Q_sink']
        Q_out = inputs['Q_out']
        m = inputs['m']
        m_flow = inputs['m_flow']
        m_burn = inputs['m_burn']

        # The change in fuel mass is the difference between the amount pumped out
        # and the amount coming back in from recirculation
        outputs['m_dot'] = -m_burn

        # Both the heat energy being added to the fuel tank and the energy
        # coming from the recirculated fuel affect the temperature of the tank
        sink_term = (1 - m_burn / m_flow)
        # Need to change NANs to 0s because m_flow might be 0
        sink_term = np.nan_to_num(sink_term)

        outputs['T_dot'] = (Q_env + sink_term * Q_sink - m_flow * Q_out) / (m * self.Cv)

        outputs['m_constraint'] = m_burn - m_flow

    # def compute_partials(self, inputs, partials):
    #     partials['T_dot', 'T_in'] = inputs['m_in'] / inputs['m']
    #     partials['T_dot', 'T'] = -inputs['m_in'] / inputs['m']
    #     partials['T_dot', 'm_in'] = (inputs['T_in'] - inputs['T']) / inputs['m']
    #     partials['T_dot', 'm'] = (inputs['T'] - inputs['T_in']) * inputs['m_in'] / inputs['m']**2 - self.q / (inputs['m']**2 * self.Cv)
