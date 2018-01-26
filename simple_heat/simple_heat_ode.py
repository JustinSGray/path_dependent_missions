from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Group, IndepVarComp

from pointer.ode_function import ODEFunction
from path_dependent_missions.simple_heat.tank_comp import TankComp
from path_dependent_missions.simple_heat.heat_exchanger_comp import HeatExchangerComp


class SimpleHeatODE(ODEFunction):

    def __init__(self):
        super(SimpleHeatODE, self).__init__(system_class=SimpleHeatSystem)

        self.declare_time(units='s')

        self.declare_state('m', units='kg', rate_source='m_dot', targets=['m'])
        self.declare_state('T', units='K', rate_source='T_dot', targets=['T'])

        self.declare_parameter('m_flow', targets=['m_flow'], units='kg/s')


class SimpleHeatSystem(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_subsystem(name='tank',
                           subsys=TankComp(num_nodes=nn),
                           promotes=['*'])

        self.add_subsystem(name='heat_exchanger',
                           subsys=HeatExchangerComp(num_nodes=nn),
                           promotes=['*'])
