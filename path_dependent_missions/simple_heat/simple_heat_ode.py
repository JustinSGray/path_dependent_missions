from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, IndepVarComp, NonlinearBlockGS, NewtonSolver, DenseJacobian, DirectSolver

from dymos import ODEOptions

from path_dependent_missions.simple_heat.tank_comp import TankComp
from path_dependent_missions.simple_heat.heat_exchanger_comp import HeatExchangerComp
from path_dependent_missions.simple_heat.fuel_burner_comp import FuelBurnerComp
from path_dependent_missions.simple_heat.power_comp import PowerComp


class SimpleHeatODE(Group):
    """
    Defines the ODE for the fuel circulation problem.
    Here we define the states and parameters (controls) for the problem.

    m : mass of the fuel in the tank
    T : temperature of the fuel in the tank
    energy : energy required to pump the fuel in the system
    """

    ode_options = ODEOptions()

    ode_options.declare_time(units='s')

    ode_options.declare_state('m', units='kg', rate_source='m_dot', targets=['m'])
    ode_options.declare_state('T', units='K', rate_source='T_dot', targets=['T'])
    ode_options.declare_state('energy', units='J', rate_source='power')

    ode_options.declare_parameter('m_flow', targets=['m_flow'], units='kg/s')
    ode_options.declare_parameter('m_burn', targets=['m_burn'], units='kg/s')

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('q_tank', types=float)
        self.metadata.declare('q_hx1', types=float)
        self.metadata.declare('q_hx2', types=float)

    def setup(self):
        nn = self.metadata['num_nodes']
        q_tank = self.metadata['q_tank']
        q_hx1 = self.metadata['q_hx1']
        q_hx2 = self.metadata['q_hx2']

        self.add_subsystem(name='tank',
                           subsys=TankComp(num_nodes=nn, q=q_tank),
                           promotes=['m', 'm_flow', 'm_dot', 'T', 'T_dot'])

        self.add_subsystem(name='heat_exchanger_pre',
                           subsys=HeatExchangerComp(num_nodes=nn, q=q_hx1),
                           promotes=[])

        self.add_subsystem(name='fuel_burner',
                           subsys=FuelBurnerComp(num_nodes=nn),
                           promotes=['m_burn'])

        self.add_subsystem(name='heat_exchanger_post',
                           subsys=HeatExchangerComp(num_nodes=nn, q=q_hx2),
                           promotes=[])

        self.add_subsystem(name='power',
                           subsys=PowerComp(num_nodes=nn),
                           promotes=['m_flow', 'power'])

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

        # Set solvers
        self.nonlinear_solver = NonlinearBlockGS()
        self.linear_solver = DirectSolver()
        self.jacobian = DenseJacobian()

if __name__ == "__main__":
    from openmdao.api import Problem, view_model, IndepVarComp

    nn = 3

    p = Problem(model=Group())

    p.model.add_subsystem('SHS', SimpleHeatSystem(num_nodes=nn), promotes=['*'])

    ivc = IndepVarComp()
    ivc.add_output('m_flow', shape=nn, val=np.linspace(0, 2, nn))
    p.model.add_subsystem('ivc', ivc, promotes=['*'])

    p.setup(check=True)
    p.run_model()
    p.check_partials(compact_print=True)

    # view_model(p)
