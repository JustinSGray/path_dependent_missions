from __future__ import print_function, division, absolute_import

from openmdao.api import Group

from dymos import ODEOptions

# from dymos.models.atmosphere import StandardAtmosphereGroup
from path_dependent_missions.escort.aero import AeroGroup
from path_dependent_missions.escort.prop.F110_prop import PropGroup
from path_dependent_missions.escort.atmos.atmos_comp import AtmosComp as StandardAtmosphereGroup
# from path_dependent_missions.escort.prop import PropGroup
from dymos.models.eom import FlightPathEOM2D


class MinTimeClimbODE(Group):

    ode_options = ODEOptions()

    ode_options.declare_time(units='s')

    ode_options.declare_state('r', units='m', rate_source='flight_dynamics.r_dot')
    ode_options.declare_state('h', units='m', rate_source='flight_dynamics.h_dot', targets=['h'])
    ode_options.declare_state('v', units='m/s', rate_source='flight_dynamics.v_dot', targets=['v'])
    ode_options.declare_state('gam', units='rad', rate_source='flight_dynamics.gam_dot',
                              targets=['gam'])
    ode_options.declare_state('m', units='kg', rate_source='prop.m_dot', targets=['m'])

    ode_options.declare_parameter('alpha', targets=['alpha'], units='rad')
    ode_options.declare_parameter('S', targets=['S'], units='m**2')
    ode_options.declare_parameter('Isp', targets=['Isp'], units='s')
    ode_options.declare_parameter('throttle', targets=['throttle'], units=None)

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

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
                           promotes_inputs=['h', 'throttle'])
                           # promotes_inputs=['h', 'throttle', 'Isp'])

        self.connect('aero.mach', 'prop.mach')

        self.add_subsystem(name='flight_dynamics',
                           subsys=FlightPathEOM2D(num_nodes=nn),
                           promotes_inputs=['m', 'v', 'gam', 'alpha'])

        self.connect('aero.f_drag', 'flight_dynamics.D')
        self.connect('aero.f_lift', 'flight_dynamics.L')
        self.connect('prop.thrust', 'flight_dynamics.T')


if __name__ == '__main__':
    from openmdao.api import Problem, view_model

    prob = Problem()

    prob.model.add_subsystem('ode', MinTimeClimbODE(num_nodes=10), promotes=None)

    prob.setup()
    view_model(prob)
