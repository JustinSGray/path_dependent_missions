from __future__ import absolute_import

import numpy as np

from openmdao.api import Group, DenseJacobian, IndepVarComp

from openaerostruct.geometry.inputs_group import InputsGroup
from openaerostruct.aerodynamics.vlm_preprocess_group import VLMPreprocessGroup
from openaerostruct.aerodynamics.vlm_states_group import VLMStatesGroup
from openaerostruct.aerodynamics.vlm_postprocess_group import VLMPostprocessGroup

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
        self.metadata.declare('num_nodes', types=int,
                              desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        num_points_x = 2
        num_points_z_half = 2
        num_points_z = 2 * num_points_z_half - 1

        wing = LiftingSurface('wing')

        wing.initialize_mesh(num_points_x, num_points_z_half, airfoil_x=np.linspace(0., 1., num_points_x), airfoil_y=np.zeros(num_points_x))
        wing.set_mesh_parameters(distribution='sine', section_origin=.25)
        wing.set_structural_properties(E=70.e9, G=29.e9, spar_location=0.35, sigma_y=200e6, rho=2700)
        wing.set_aero_properties(factor2=.119, factor4=-0.064, cl_factor=1.05, CD0=0.02)

        wing.set_chord(1., n_cp=3, order=2)
        wing.set_twist(0., n_cp=3, order=2)
        wing.set_sweep(0., n_cp=3, order=2)
        wing.set_dihedral(0., n_cp=3, order=2)
        wing.set_span(15., n_cp=3, order=2)
        wing.set_thickness(0.05)
        wing.set_radius(0.1)

        lifting_surfaces = [('wing', wing)]

        vlm_scaler = 1e0

        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('Re_1e6', shape=num_nodes, val=2.)
        indep_var_comp.add_output('C_l_max', shape=num_nodes, val=1.5)
        indep_var_comp.add_output('reference_area', shape=num_nodes, val=484.)
        self.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

        inputs_group = InputsGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('inputs_group', inputs_group, promotes=['*'])

        self.add_subsystem('vlm_preprocess_group',
            VLMPreprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'])

        self.add_subsystem('vlm_states_group',
            VLMStatesGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vlm_scaler=vlm_scaler), promotes=['*'])

        self.add_subsystem('vlm_postprocess_group',
            VLMPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
            promotes=['*'])
