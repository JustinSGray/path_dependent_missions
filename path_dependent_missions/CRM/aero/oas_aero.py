from __future__ import absolute_import

import numpy as np

from openmdao.api import Group, IndepVarComp

from openaerostruct.geometry.inputs_group import InputsGroup
from openaerostruct.aerodynamics.vlm_full_group import VLMFullGroup

from openaerostruct.common.lifting_surface import LiftingSurface


class OASGroup(Group):
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
        num_nodes = self.options['num_nodes']
        num_points_x = 2
        num_points_z = 3

        wing = LiftingSurface('wing')

        wing.initialize_mesh(num_points_x, num_points_z, distribution='sine', section_origin=.25)
        wing.set_structural_properties(E=70.e9, G=29.e9, spar_location=0.35, material_yield=200e6, material_density=2700)

        wing.set_chord(1.)
        wing.set_twist(0.)
        wing.set_displacement_x(np.zeros(5))
        wing.set_displacement_y(np.zeros(5))
        wing.set_displacement_z(np.zeros(5))
        wing.set_thickness(0.05)
        wing.set_radius(0.1)

        wing.bsplines['chord'] = [5, 2]
        wing.bsplines['twist'] = [5, 2]
        wing.bsplines['sec_x'] = [5, 2]
        wing.bsplines['sec_y'] = [5, 2]
        wing.bsplines['sec_z'] = [5, 2]

        lifting_surfaces = [wing]

        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('Re_1e6', shape=num_nodes, val=2.)
        indep_var_comp.add_output('C_l_max', shape=num_nodes, val=1.5)
        indep_var_comp.add_output('reference_area', shape=num_nodes, val=484.)
        self.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

        inputs_group = InputsGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('inputs_group', inputs_group, promotes=['*'])

        self.add_subsystem('vlm_full_group',
            VLMFullGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'])
