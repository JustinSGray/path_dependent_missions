from __future__ import absolute_import

from openmdao.api import Group

from .mdot_comp import MassFlowRateComp
from .smt_thrust_throttle import SMTThrustComp


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


        self.add_subsystem(name='thrust_comp',
                           subsys=SMTThrustComp(num_nodes=nn),
                           promotes_inputs=['mach', 'h', 'throttle'],
                           promotes_outputs=['thrust', 'm_dot'])
