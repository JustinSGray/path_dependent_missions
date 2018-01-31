from __future__ import absolute_import

from openmdao.api import Group

from .mdot_comp import MassFlowRateComp
from .bryson_thrust_comp import BrysonThrustComp
from .mbi_thrust_comp import MBIThrustComp


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

        self.metadata.declare('mbi_thrust', types=bool, default=False,
                              desc='If True, use MBI interpolant for thrust comp')

    def setup(self):
        nn = self.metadata['num_nodes']

        if self.metadata['mbi_thrust']:
            thrust_comp = MBIThrustComp(num_nodes=nn)
        else:
            thrust_comp = BrysonThrustComp(num_nodes=nn)

        self.add_subsystem(name='thrust_comp',
                           subsys=thrust_comp,
                           promotes_inputs=['mach', 'h'],
                           promotes_outputs=['thrust'])

        self.add_subsystem(name='mdot_comp',
                           subsys=MassFlowRateComp(num_nodes=nn),
                           promotes_inputs=['thrust', 'Isp'],
                           promotes_outputs=['m_dot'])
