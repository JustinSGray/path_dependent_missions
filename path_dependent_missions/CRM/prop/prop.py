from __future__ import absolute_import

from openmdao.api import Group

from .smt_thrust_comp import SMTMaxThrustComp
from .thrust_comp import ThrustComp
from .SFC_comp import SFCComp
from .fuel_rate_comp import FuelRateComp

from path_dependent_missions.CRM.prop.b777_engine_data import get_prop_smt_model


class PropGroup(Group):
    """
    The purpose of the PropGroup is to compute the propulsive forces on the
    aircraft in the body frame.

    Parameters
    ----------
    mach : float
        Mach number (unitless)
    alt : float
        altitude (m)
    Isp : float
        specific impulse (s)
    throttle : float
        throttle value nominally between 0.0 and 1.0 (unitless)

    Unknowns
    --------
    thrust : float
        Vehicle thrust force (N)
    mdot : float
        Vehicle mass accumulation rate (kg/s)

    """
    def initialize(self):
        self.metadata.declare('num_nodes', types=int,
                              desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.metadata['num_nodes']

        smt_prop_model = get_prop_smt_model()
        max_thrust_comp = SMTMaxThrustComp(num_nodes=nn, propulsion_model=smt_prop_model)

        self.add_subsystem(name='max_thrust_comp',
                           subsys=max_thrust_comp,
                           promotes_inputs=['mach', 'h'],
                           promotes_outputs=['max_thrust'])

        self.add_subsystem(name='thrust_comp',
                           subsys=ThrustComp(num_nodes=nn),
                           promotes_inputs=['max_thrust', 'throttle'],
                           promotes_outputs=['thrust'])

        self.add_subsystem(name='sfc_comp',
                           subsys=SFCComp(num_nodes=nn, propulsion_model=smt_prop_model),
                           promotes_inputs=['mach', 'h', 'throttle'],
                           promotes_outputs=['SFC_1em6_NNs'])

        self.add_subsystem(name='fuel_rate_comp',
                           subsys=FuelRateComp(num_nodes=nn),
                           promotes_inputs=['thrust', 'SFC_1em6_NNs'],
                           promotes_outputs=['m_dot'])
