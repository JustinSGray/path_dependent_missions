from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, IndepVarComp, NonlinearBlockGS, NewtonSolver

from pointer.ode_function import ODEFunction
from path_dependent_missions.simple_heat.tank_comp import TankComp
from path_dependent_missions.simple_heat.heat_exchanger_comp import HeatExchangerComp
from path_dependent_missions.simple_heat.fuel_burner_comp import FuelBurnerComp


class SimpleHeatODE(ODEFunction):

    def __init__(self):
        super(SimpleHeatODE, self).__init__(system_class=SimpleHeatSystem)

        self.declare_time(units='s')

        self.declare_state('m', units='kg', rate_source='m_dot', targets=['m'])
        self.declare_state('T', units='K', rate_source='T_dot', targets=['T'])

        self.declare_parameter('m_flow', targets=['m_flow'], units='kg/s')
        self.declare_parameter('m_burn', targets=['m_burn'], units='kg/s')


class SimpleHeatSystem(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_subsystem(name='tank',
                           subsys=TankComp(num_nodes=nn),
                           promotes=['m', 'm_flow', 'm_dot', 'T', 'T_dot'])

        self.add_subsystem(name='heat_exchanger_pre',
                           subsys=HeatExchangerComp(num_nodes=nn, q=15.),
                           promotes=[])

        self.add_subsystem(name='fuel_burner',
                           subsys=FuelBurnerComp(num_nodes=nn),
                           promotes=['m_burn'])

        self.add_subsystem(name='heat_exchanger_post',
                           subsys=HeatExchangerComp(num_nodes=nn, q=-5.),
                           promotes=[])

        # Tank to HX1
        self.connect('tank.T_out', 'heat_exchanger_pre.T_in')
        self.connect('tank.m_out', 'heat_exchanger_pre.m_in')

        # HX1 to HX2
        self.connect('heat_exchanger_pre.T_out', 'heat_exchanger_post.T_in')

        # HX1 to burner
        self.connect('tank.m_out', 'fuel_burner.m_in')

        # Burner to HX2
        self.connect('fuel_burner.m_recirculated', 'heat_exchanger_post.m_in')

        # HX2 to tank
        self.connect('heat_exchanger_post.T_out', 'tank.T_in')
        self.connect('fuel_burner.m_recirculated', 'tank.m_in')

        self.nonlinear_solver = NonlinearBlockGS()
        self.nonlinear_solver.options['iprint'] = 2

if __name__ == "__main__":
    from openmdao.api import Problem, view_model

    p = Problem()

    p.model = SimpleHeatSystem(num_nodes=5)

    p.setup(check=True)
    p.run_model()
    p.check_partials(compact_print=True)

    # view_model(p)
