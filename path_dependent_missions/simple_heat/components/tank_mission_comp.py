import numpy as np
from openmdao.api import ExplicitComponent


class TankMissionComp(ExplicitComponent):
    """
    This contains information about the fuel tank, especially its temperature,
    mass of fuel, flow in and out, and how those all change.

    This is using the simplified equation from Alyanak's paper.
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        self.nn = self.metadata['num_nodes']

        self.add_input('m_fuel', shape=self.nn, units='kg', val=1.)
        self.add_input('T', shape=self.nn, units='K')
        self.add_input('m_flow', val=1., shape=self.nn, units='kg/s')
        self.add_input('m_burn', val=1., shape=self.nn, units='kg/s')
        self.add_input('Q_env', shape=self.nn, units='W')
        self.add_input('Q_sink', shape=self.nn, units='W')
        self.add_input('Q_out', shape=self.nn, units='W')
        self.add_input('Cv', shape=self.nn, units='J/(kg*K)')

        self.add_output('T_dot', shape=self.nn, units='K/s')
        self.add_output('T_o', shape=self.nn, units='K')
        self.add_output('m_recirculated', shape=self.nn, units='kg/s')

        self.ar = ar = np.arange(self.nn)

        self.declare_partials('T_dot', 'Q_env', rows=ar, cols=ar)
        self.declare_partials('T_dot', 'Q_out', rows=ar, cols=ar)
        self.declare_partials('T_dot', 'Q_sink', rows=ar, cols=ar)
        self.declare_partials('T_dot', 'm_fuel', rows=ar, cols=ar)
        self.declare_partials('T_dot', 'm_burn', rows=ar, cols=ar)
        self.declare_partials('T_dot', 'm_flow', rows=ar, cols=ar)
        self.declare_partials('T_dot', 'Cv', rows=ar, cols=ar)

        self.declare_partials('m_recirculated', 'm_burn', rows=ar, cols=ar, val=-1.)
        self.declare_partials('m_recirculated', 'm_flow', rows=ar, cols=ar, val=1.)

        self.declare_partials('T_o', 'T', rows=ar, cols=ar, val=1.)
        self.declare_partials('T_o', 'Q_sink', rows=ar, cols=ar)
        self.declare_partials('T_o', 'm_flow', rows=ar, cols=ar)
        self.declare_partials('T_o', 'Cv', rows=ar, cols=ar)

        # self.declare_partials('*', '*', method='cs')
        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        Q_env = inputs['Q_env']
        Q_sink = inputs['Q_sink']
        Q_out = inputs['Q_out']
        m_fuel = inputs['m_fuel']
        m_flow = inputs['m_flow']
        m_burn = inputs['m_burn']
        Cv = inputs['Cv']

        # Both the heat energy being added to the fuel tank and the energy
        # coming from the recirculated fuel affect the temperature of the tank
        sink_term = (1 - m_burn / m_flow)
        # Need to change NANs to 0s because m_flow might be 0
        sink_term = np.nan_to_num(sink_term)

        outputs['T_dot'] = (Q_env + sink_term * Q_sink - (m_flow - m_burn) * Q_out) / (m_fuel * Cv)

        outputs['m_recirculated'] = m_flow - m_burn

        outputs['T_o'] = inputs['T'] + Q_sink / (m_flow * Cv)

    def compute_partials(self, inputs, partials):
        Q_env = inputs['Q_env']
        Q_sink = inputs['Q_sink']
        Q_out = inputs['Q_out']
        m_fuel = inputs['m_fuel']
        m_flow = inputs['m_flow']
        m_burn = inputs['m_burn']
        Cv = inputs['Cv']

        # Both the heat energy being added to the fuel tank and the energy
        # coming from the recirculated fuel affect the temperature of the tank
        sink_term = (1 - m_burn / m_flow)
        # Need to change NANs to 0s because m_flow might be 0
        sink_term = np.nan_to_num(sink_term)

        partials['T_dot', 'Q_env'] = 1. / (m_fuel * Cv)
        partials['T_dot', 'Q_sink'] = sink_term / (m_fuel * Cv)
        partials['T_dot', 'Q_out'] = - (m_flow - m_burn) / (m_fuel * Cv)
        partials['T_dot', 'Cv'] = - (Q_env + sink_term * Q_sink - (m_flow - m_burn) * Q_out) / (m_fuel * Cv**2)

        partials['T_dot', 'm_fuel'] = - (Q_env + sink_term * Q_sink - (m_flow - m_burn) * Q_out) / (m_fuel**2 * Cv)
        partials['T_dot', 'm_burn'] = (- Q_sink / m_flow + Q_out) / (m_fuel * Cv)
        partials['T_dot', 'm_flow'] = (m_burn / m_flow**2 * Q_sink - Q_out) / (m_fuel * Cv)

        partials['T_o', 'Q_sink'] = 1. / (m_flow * Cv)
        partials['T_o', 'm_flow'] = - Q_sink / (m_flow**2 * Cv)
        partials['T_o', 'Cv'] = - Q_sink / (m_flow * Cv**2)
