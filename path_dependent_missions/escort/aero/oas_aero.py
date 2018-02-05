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
        lifting_surfaces = get_default_lifting_surfaces()

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
