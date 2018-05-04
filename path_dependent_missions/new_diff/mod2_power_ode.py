from __future__ import absolute_import

import numpy as np

from openmdao.api import Group, DirectSolver, NewtonSolver, BalanceComp, IndepVarComp, BoundsEnforceLS, \
                         DenseJacobian, ArmijoGoldsteinLS

from dymos import declare_time, declare_state, declare_parameter

from path_dependent_missions.escort.atmos.atmos_comp import AtmosComp
from path_dependent_missions.escort.aero import AeroGroup
from path_dependent_missions.escort.prop.F110_prop import PropGroup
from tas_rate_comp import TASRateComp
from unsteady_flight_dynamics_comp import UnsteadyFlightDynamicsComp
from pass_through_comp import PassThroughComp

@declare_time(units='s')
@declare_state('m', targets=['flight_equilibrium_group.dynamics.m'],
                    rate_source='flight_equilibrium_group.prop.m_dot', units='kg')
@declare_state('h', targets=['atmos.h', 'flight_equilibrium_group.prop.h'],
                    rate_source='pass_through.h_dot_out', units='m')
@declare_state('r', rate_source='pass_through.r_dot_out',
                    units='m')
@declare_state('TAS', targets=['flight_equilibrium_group.aero.v', 'flight_equilibrium_group.dynamics.TAS'], rate_source='tas_rate_comp.TAS_rate', units='m/s')
@declare_parameter('h_dot', targets=['climb_rate.h_dot'], units='m/s')
@declare_parameter('r_dot', targets=['range_rate.r'], units='m/s')
@declare_parameter('S', targets=['flight_equilibrium_group.aero.S'],
                        units='m**2')
class X57Mod2PowerODE(Group):
    def initialize(self):
        self.metadata.declare('num_nodes', types=int,
                              desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_subsystem(name='atmos',
                           subsys=AtmosComp(num_nodes=nn))

        self.add_subsystem(name='pass_through', subsys=PassThroughComp(num_nodes=nn))

        self.add_subsystem(name='tas_rate_comp', subsys=TASRateComp(num_nodes=nn))

        flight_equilibrium_group = self.add_subsystem('flight_equilibrium_group', subsys=Group())

        flight_equilibrium_group.add_subsystem('aero', subsys=AeroGroup(num_nodes=nn))

        flight_equilibrium_group.add_subsystem(name='prop', subsys=PropGroup(num_nodes=nn))

        flight_equilibrium_group.add_subsystem('dynamics',
                                            subsys=UnsteadyFlightDynamicsComp(num_nodes=nn))

        alpha_bal = BalanceComp()
        alpha_bal.add_balance(name='alpha', val=1.0 * np.ones(nn), units='rad', rhs_name='TAS_rate',
                                     lhs_name='TAS_rate_computed', eq_units='m/s**2', lower=-5., upper=5.)
        alpha_bal.add_balance(name='throttle', val=np.ones(nn), units=None, rhs_name='gam_rate',
                                     lhs_name='gam_rate_computed', eq_units='rad/s', lower=0., upper=1.)

        flight_equilibrium_group.add_subsystem(name='alpha_bal', subsys=alpha_bal)

        self.connect('atmos.rho', 'flight_equilibrium_group.aero.rho')
        self.connect('flight_equilibrium_group.dynamics.TAS_rate_computed', 'flight_equilibrium_group.alpha_bal.TAS_rate_computed')
        self.connect('flight_equilibrium_group.dynamics.gam_rate_computed', 'flight_equilibrium_group.alpha_bal.gam_rate_computed')

        self.connect('flight_equilibrium_group.aero.f_lift', 'flight_equilibrium_group.dynamics.L')
        self.connect('flight_equilibrium_group.aero.f_drag', 'flight_equilibrium_group.dynamics.D')
        self.connect('flight_equilibrium_group.alpha_bal.alpha', ('flight_equilibrium_group.dynamics.alpha', 'flight_equilibrium_group.aero.alpha'))
        self.connect('flight_equilibrium_group.aero.mach', 'flight_equilibrium_group.prop.mach')
        self.connect('flight_equilibrium_group.prop.thrust', 'flight_equilibrium_group.dynamics.thrust')


        self.connect('tas_rate_comp.TAS_dot', 'flight_equilibrium_group.alpha_bal.TAS_rate')
        self.connect('tas_rate_comp.gam_dot', 'flight_equilibrium_group.alpha_bal.gam_rate')
        self.connect('tas_rate_comp.gam', 'flight_equilibrium_group.dynamics.gam')

        self.connect('flight_equilibrium_group.alpha_bal.throttle', 'flight_equilibrium_group.prop.throttle')
        self.connect('atmos.sos', 'flight_equilibrium_group.aero.sos')

        self.connect("pass_through.h_dot_out", "tas_rate_comp.h_dot")
        self.connect("pass_through.r_dot_out", "tas_rate_comp.r_dot")

        flight_equilibrium_group.linear_solver = DirectSolver()
        # flight_equilibrium_group.linear_solver = ScipyKrylov()

        flight_equilibrium_group.nonlinear_solver = NewtonSolver()
        # self.jacobian = DenseJacobian()
        # flight_equilibrium_group.nonlinear_solver.linesearch = BoundsEnforceLS()
        # flight_equilibrium_group.nonlinear_solver.linesearch.options['bound_enforcement'] = 'vector'
        flight_equilibrium_group.nonlinear_solver.linesearch = ArmijoGoldsteinLS()
        flight_equilibrium_group.nonlinear_solver.options['solve_subsystems'] = True
        flight_equilibrium_group.nonlinear_solver.linesearch.options['print_bound_enforce'] = True
        # flight_equilibrium_group.nonlinear_solver.options['err_on_maxiter'] = True
        flight_equilibrium_group.nonlinear_solver.options['max_sub_solves'] = 20    # default=10
        flight_equilibrium_group.nonlinear_solver.options['debug_print'] = True
        flight_equilibrium_group.nonlinear_solver.options['iprint'] = 2  # default=10
        flight_equilibrium_group.nonlinear_solver.options['maxiter'] = 20    # default=10
        # flight_equilibrium_group.nonlinear_solver.linesearch.options['print_bound_enforce'] = True

        # throttle_equilibrium_group.nonlinear_solver = NonlinearBlockGS()
        # flight_equilibrium_group.nonlinear_solver.options['maxiter'] = 1

if __name__ == "__main__":
    from openmdao.api import Problem, view_model, IndepVarComp
    from numpy import array

    nn = 1

    p = Problem(model=Group())

    p.model.add_subsystem('DIODE', X57Mod2PowerODE(num_nodes=nn), promotes=['*'])

    p.setup(check=True, force_alloc_complex=True)

    # for var, val in inputs.items():
    #     p[var][0] = val[0]
    #
    # for var, val in outputs.items():
    #     p[var][0] = val[0]

    p['DIODE.flight_equilibrium_group.alpha_bal.alpha'] = -0.1

    from openmdao.api import view_model
    view_model(p)

    p.run_model()

    p.model.list_inputs(print_arrays=True)
    p.model.list_outputs(print_arrays=True, residuals=True)
