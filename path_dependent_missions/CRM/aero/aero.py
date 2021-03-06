from __future__ import absolute_import

import numpy as np

from openmdao.api import Group

from .mach_comp import MachComp
from path_dependent_missions.CRM.aero.oas_aero import OASGroup


class AeroGroup(Group):
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

        self.add_subsystem(name='OAS_group',
                           subsys=OASGroup(num_nodes=nn),
                           promotes_inputs=[('rho', 'rho'), ('v', 'v'), 'alpha'],
                           promotes_outputs=[('lift', 'f_lift'), ('drag', 'f_drag'),
                               ('C_L', 'CL'), ('C_D', 'CD')],
                           )
