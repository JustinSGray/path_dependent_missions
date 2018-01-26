from openmdao.api import ExplicitComponent


class HeatExchangerComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        self.nn = self.metadata['num_nodes']

        self.add_input('T', shape=self.nn, units='K')
        self.add_input('m_dot', shape=self.nn, units='kg/s')
        self.add_output('T_0', shape=self.nn, units='K')

        self.q_sink = 2.
        self.Cv = 1.

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        T_0 = self.q_sink / (inputs['m_dot'] * self.Cv) + inputs['T']
        outputs['T_0'] = T_0
