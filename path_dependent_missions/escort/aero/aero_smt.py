from __future__ import absolute_import

import numpy as np

from openmdao.api import Group

from .dynamic_pressure_comp import DynamicPressureComp
from .lift_drag_force_comp import LiftDragForceComp
from .cd0_comp import CD0Comp
from .kappa_comp import KappaComp
from .cla_comp import CLaComp
from .cl_comp import CLComp
from .cd_comp import CDComp
from .mach_comp import MachComp
from .aero_smt_comp import AeroSMTComp


class AeroSMTGroup(Group):
    """
    The purpose of the AeroGroup is to compute the aerodynamic forces on the
    aircraft in the body frame.

    Parameters
    ----------
    v : float
        air-relative velocity (m/s)
    sos : float
        local speed of sound (m/s)
    rho : float
        atmospheric density (kg/m**3)
    alpha : float
        angle of attack (rad)
    S : float
        aerodynamic reference area (m**2)

    """
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                              desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='mach_comp',
                           subsys=MachComp(num_nodes=nn),
                           promotes_inputs=['v', 'sos'],
                           promotes_outputs=['mach'])

        self.add_subsystem(name='SMT_comp',
                           subsys=AeroSMTComp(num_nodes=nn),
                           promotes_inputs=['mach', 'h', 'alpha'],
                           promotes_outputs=['CL', 'CD'])

        self.add_subsystem(name='q_comp',
                           subsys=DynamicPressureComp(num_nodes=nn),
                           promotes_inputs=['rho', 'v'],
                           promotes_outputs=['q'])

        self.add_subsystem(name='lift_drag_force_comp',
                           subsys=LiftDragForceComp(num_nodes=nn),
                           promotes_inputs=['CL', 'CD', 'q', 'S'],
                           promotes_outputs=['f_lift', 'f_drag'])
