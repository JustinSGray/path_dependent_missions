from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, IndepVarComp

from pointer.ode_function import ODEFunction

from pointer.models.atmosphere import StandardAtmosphereGroup
from .aero import AeroGroup
from .prop import PropGroup
from pointer.models.eom import FlightPathEOM2D


class MinTimeClimbODE(ODEFunction):

    def __init__(self, thrust_model='bryson'):
        super(MinTimeClimbODE, self).__init__(system_class=BrysonMinTimeClimbSystem,
                                              system_init_kwargs={'thrust_model': thrust_model})

        self.declare_time(units='s')

        self.declare_state('r', units='m', rate_source='flight_dynamics.r_dot')
        self.declare_state('h', units=,m'm', rate_source='flight_dynamics.h_dot', targets=['h'])
        self.declare_state('v', units='m/s', rate_source='flight_dynamics.v_dot', targets=['v'])
        self.declare_state('gam', units='rad', rate_source='flight_dynamics.gam_dot',
                           targets=['gam'])
        self.declare_state('m', units='kg', rate_source='prop.m_dot', targets=['m'])

        # self.declare_parameter('alpha', targets=['alpha'], units='rad')
        self.declare_parameter('alpha', targets=['alpha', 'aero.OAS_group.alpha_rad'], units='rad')

        self.declare_parameter('S', targets=['S'], units='m**2')
        self.declare_parameter('throttle', targets=['throttle'], units=None)


class BrysonMinTimeClimbSystem(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('thrust_model', types=str)

    def setup(self):
        nn = self.metadata['num_nodes']
        thrust_model = self.metadata['thrust_model']

        # # We'll use an IndepVarComp in the phase to provide Isp and reference area.
        # ivc = IndepVarComp()
        # ivc.add_output(name='Isp', val=1600*np.ones(nn), units='s')
        # ivc.add_output(name='S', val=49.2386*np.ones(nn), units='m**2')
        #
        # self.connect('S', 'aero.S')
        # self.connect('Isp', 'prop.Isp')
        #
        # self.add_subsystem(name='ivc',
        #                    subsys=ivc,
        #                    promotes_outputs=['*'])

        self.add_subsystem(name='atmos',
                           subsys=StandardAtmosphereGroup(num_nodes=nn),
                           promotes_inputs=['h'])

        self.add_subsystem(name='aero',
                           subsys=AeroGroup(num_nodes=nn),
                           # promotes_inputs=['v', 'alpha', 'S'],
                           promotes_inputs=['v', 'S'],
                           )

        self.connect('atmos.sos', 'aero.sos')
        self.connect('atmos.rho', 'aero.rho')

        self.add_subsystem(name='prop',
                           subsys=PropGroup(num_nodes=nn, thrust_model=thrust_model),
                           promotes_inputs=['h', 'throttle'])

        self.connect('aero.mach', 'prop.mach')

        self.add_subsystem(name='flight_dynamics',
                           subsys=FlightPathEOM2D(num_nodes=nn),
                           promotes_inputs=['m', 'v', 'gam', 'alpha'])

        self.connect('aero.f_drag', 'flight_dynamics.D')
        self.connect('aero.f_lift', 'flight_dynamics.L')
        self.connect('prop.thrust', 'flight_dynamics.T')
