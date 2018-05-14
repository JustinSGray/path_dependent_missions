from __future__ import print_function, division, absolute_import
import numpy as np

from openmdao.api import Group, IndepVarComp, NonlinearBlockGS, NewtonSolver, DenseJacobian, DirectSolver, CSCJacobian, CSRJacobian, ExecComp

from dymos import ODEOptions

from path_dependent_missions.escort.aero import AeroGroup
from path_dependent_missions.escort.aero.aero_smt import AeroSMTGroup
from path_dependent_missions.escort.prop.F110_prop import PropGroup
from path_dependent_missions.escort.atmos.atmos_comp import AtmosComp as StandardAtmosphereGroup
from dymos.models.eom import FlightPathEOM2D
from path_dependent_missions.simple_heat.components.tank_mission_comp import TankMissionComp
from path_dependent_missions.simple_heat.components.power_comp import PowerComp
from path_dependent_missions.simple_heat.components.cv_comp import CvComp
from path_dependent_missions.simple_heat.components.pump_heating_comp import PumpHeatingComp
from path_dependent_missions.simple_heat.components.engine_heating_comp import EngineHeatingComp

class ThermalMissionODE(Group):

    ode_options = ODEOptions()

    ode_options.declare_time(units='s')

    # Mission and aero
    ode_options.declare_state('r', units='m', rate_source='flight_dynamics.r_dot')
    ode_options.declare_state('h', units='m', rate_source='flight_dynamics.h_dot', targets=['h'])
    ode_options.declare_state('v', units='m/s', rate_source='flight_dynamics.v_dot', targets=['v'])
    ode_options.declare_state('gam', units='rad', rate_source='flight_dynamics.gam_dot',
                              targets=['gam'])
    ode_options.declare_state('m', units='kg', rate_source='m_dot', targets=['m'])

    ode_options.declare_parameter('alpha', targets=['alpha'], units='rad')
    ode_options.declare_parameter('S', targets=['S'], units='m**2')
    ode_options.declare_parameter('throttle', targets=['throttle'], units=None)
    ode_options.declare_parameter('W0', targets=['W0'], units='kg')

    # Thermal
    ode_options.declare_state('T', units='K', rate_source='T_dot', targets=['T'])
    ode_options.declare_state('energy', units='J', rate_source='power')

    ode_options.declare_parameter('m_recirculated', targets=['m_recirculated'], units='kg/s')
    ode_options.declare_parameter('Q_env', targets=['Q_env'], units='W')
    ode_options.declare_parameter('Q_sink', targets=['Q_sink'], units='W')
    ode_options.declare_parameter('Q_out', targets=['Q_out'], units='W')

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('engine_heat_coeff', types=float)
        self.options.declare('pump_heat_coeff', types=float)

    def setup(self):
        nn = self.options['num_nodes']
        engine_heat_coeff = self.options['engine_heat_coeff']
        pump_heat_coeff = self.options['pump_heat_coeff']

        # Aero and mission
        self.add_subsystem(name='atmos',
                           subsys=StandardAtmosphereGroup(num_nodes=nn),
                           promotes_inputs=['h'])

        self.add_subsystem(name='aero',
                           subsys=AeroGroup(num_nodes=nn),
                           promotes_inputs=['v', 'alpha', 'S'])

        self.connect('atmos.sos', 'aero.sos')
        self.connect('atmos.rho', 'aero.rho')

        self.add_subsystem(name='prop',
                           subsys=PropGroup(num_nodes=nn),
                           promotes_inputs=['h', 'throttle'],
                           promotes_outputs=['m_dot'])

        self.connect('aero.mach', 'prop.mach')

        self.add_subsystem(name='flight_dynamics',
                           subsys=FlightPathEOM2D(num_nodes=nn),
                           promotes_inputs=['m', 'v', 'gam', 'alpha'])

        self.connect('aero.f_drag', 'flight_dynamics.D')
        self.connect('aero.f_lift', 'flight_dynamics.L')
        self.connect('prop.thrust', 'flight_dynamics.T')

        # Thermal
        self.add_subsystem('m_burn_comp',
            ExecComp('m_burn = - m_dot', m_burn=np.zeros(nn), m_dot=np.zeros(nn)),
            promotes=['*'],
        )

        self.add_subsystem('m_fuel_comp',
            ExecComp('m_fuel = m - W0', m_fuel=np.zeros(nn), m=np.zeros(nn), W0=np.zeros(nn)),
            promotes=['*'],
        )

        self.add_subsystem('m_flow_comp',
            ExecComp('m_flow = m_burn + m_recirculated', m_flow=np.zeros(nn), m_burn=np.zeros(nn), m_recirculated=np.zeros(nn)),
            promotes=['*'],
        )

        self.add_subsystem(name='pump_heating_comp',
                           subsys=PumpHeatingComp(num_nodes=nn, heat_coeff=pump_heat_coeff),
                           promotes_inputs=['m_flow'],
                           promotes_outputs=['Q_pump'])

        self.add_subsystem(name='engine_heating_comp',
                          subsys=EngineHeatingComp(num_nodes=nn, heat_coeff=engine_heat_coeff),
                          promotes_inputs=['throttle'],
                          promotes_outputs=['Q_engine'])

        self.add_subsystem('Q_env_tot_comp',
            ExecComp('Q_env_tot = Q_env + Q_pump + Q_engine', Q_env_tot=np.zeros(nn), Q_env=np.zeros(nn), Q_pump=np.zeros(nn), Q_engine=np.zeros(nn)),
            promotes=['*'],
        )

        self.add_subsystem(name='cv',
                           subsys=CvComp(num_nodes=nn),
                           promotes_inputs=['T'],
                           promotes_outputs=['Cv'])

        self.add_subsystem(name='tank',
                           subsys=TankMissionComp(num_nodes=nn),
                           promotes_inputs=['m_fuel', 'm_flow', 'm_burn', 'T', 'Q_env_tot', 'Q_sink', 'Q_out', 'Cv'],
                           promotes_outputs=['T_dot', 'T_o'])

        self.add_subsystem(name='power',
                           subsys=PowerComp(num_nodes=nn),
                           promotes=['m_flow', 'power'])


        # Set solvers
        self.linear_solver = DirectSolver()
        self.jacobian = CSCJacobian()


if __name__ == '__main__':
    from openmdao.api import Problem, view_model

    prob = Problem()

    prob.model.add_subsystem('ode', ThermalMissionODE(num_nodes=3), promotes=None)

    prob.setup()
    view_model(prob)
