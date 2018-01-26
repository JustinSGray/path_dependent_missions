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

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs['T_out'] = self.q / (inputs['m_in'] * self.Cv + tol) + inputs['T_in']
