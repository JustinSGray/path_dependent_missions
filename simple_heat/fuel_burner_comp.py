from openmdao.api import ExplicitComponent


class FuelBurnerComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        self.nn = self.metadata['num_nodes']

        self.add_input('m_in', shape=self.nn, units='kg/s')
        self.add_input('m_burn', shape=self.nn, units='kg/s')
        self.add_output('m_recirculated', shape=self.nn, units='kg/s')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs['m_recirculated'] = inputs['m_in'] - inputs['m_burn']
