from __future__ import absolute_import

import numpy as np

from openmdao.api import Group, DenseJacobian, IndepVarComp

from openaerostruct.geometry.inputs_group import InputsGroup
from openaerostruct.aerodynamics.vlm_preprocess_group import VLMPreprocessGroup
from openaerostruct.aerodynamics.vlm_states_group import VLMStatesGroup
from openaerostruct.aerodynamics.vlm_postprocess_group import VLMPostprocessGroup

from openaerostruct.tests.utils import get_default_lifting_surfaces


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
        g = 9.81
        lifting_surfaces = [
            ('wing', {
                'num_points_x': num_points_x, 'num_points_z_half': num_points_z_half,
                'airfoil_x': np.linspace(0., 1., num_points_x),
                'airfoil_y': np.zeros(num_points_x),
                'chord': 1., 'twist': 0. * np.pi / 180., 'sweep_x': 0., 'dihedral_y': 0., 'span': 15,
                'twist_bspline': (6, 2),
                'sec_z_bspline': (num_points_z_half, 2),
                'chord_bspline': (2, 2),
                'thickness_bspline': (6, 3),
                'thickness' : 0.05,
                'radius' : 0.1,
                'distribution': 'sine',
                'section_origin': 0.25,
                'spar_location': 0.35,
                'E': 70.e9,
                'G': 29.e9,
                'sigma_y': 200e6,
                'rho': 2700,
                'factor2' : 0.119,
                'factor4' : -0.064,
                'cl_factor' : 1.05,
                'W0' : (0.1381 * g - .350) * 1e6 + 300 * 80 * g,
                'a' : 295.4,
                'R' : 7000. * 1.852 * 1e3,
                'M' : .84,
                'CT' : g * 17.e-6,
                'CD0' : 0.015,
                # 'CL0' : 0.2,
            })
        ]

        vlm_scaler = 1e0

        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('Re_1e6', shape=num_nodes, val=2.)
        indep_var_comp.add_output('C_l_max', shape=num_nodes, val=1.5)
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
