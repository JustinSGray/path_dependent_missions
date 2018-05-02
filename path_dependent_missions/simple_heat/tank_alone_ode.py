from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, IndepVarComp, NonlinearBlockGS, NewtonSolver, DenseJacobian, DirectSolver, CSCJacobian, CSRJacobian

from dymos import ODEOptions

from path_dependent_missions.simple_heat.components.tank_alone_comp import TankAloneComp
from path_dependent_missions.simple_heat.components.power_comp import PowerComp
from path_dependent_missions.simple_heat.components.cv_comp import CvComp


class TankAloneODE(Group):
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
    ode_options.declare_parameter('Q_env', targets=['Q_env'], units='W')
    ode_options.declare_parameter('Q_sink', targets=['Q_sink'], units='W')
    ode_options.declare_parameter('Q_out', targets=['Q_out'], units='W')

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_subsystem(name='cv',
                           subsys=CvComp(num_nodes=nn),
                           promotes_inputs=['T'],
                           promotes_outputs=['Cv'])

        self.add_subsystem(name='tank',
                           subsys=TankAloneComp(num_nodes=nn),
                           promotes_inputs=['m', 'm_flow', 'm_burn', 'T', 'Q_env', 'Q_sink', 'Q_out', 'Cv'],
                           promotes_outputs=['m_dot', 'T_dot', 'm_recirculated', 'T_o'])

        self.add_subsystem(name='power',
                           subsys=PowerComp(num_nodes=nn),
                           promotes=['m_flow', 'power'])

        # Set solvers
        self.nonlinear_solver = NonlinearBlockGS()
        self.linear_solver = DirectSolver()
        self.jacobian = CSCJacobian()

if __name__ == "__main__":
    from openmdao.api import Problem, view_model, IndepVarComp

    nn = 3

    p = Problem(model=Group())

    p.model.add_subsystem('SHS', TankAloneODE(num_nodes=nn), promotes=['*'])

    ivc = IndepVarComp()
    ivc.add_output('m_flow', shape=nn, val=np.linspace(5, 2, nn))
    ivc.add_output('m_burn', shape=nn, val=1.2)
    ivc.add_output('Q_env', shape=nn, val=1.423)
    ivc.add_output('Q_sink', shape=nn, val=24.3)
    ivc.add_output('Q_out', shape=nn, val=0.544)
    p.model.add_subsystem('ivc', ivc, promotes=['*'])

    p.setup(check=True, force_alloc_complex=True)
    p.run_model()
    p.check_partials(compact_print=True)

    # view_model(p)
