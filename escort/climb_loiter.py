from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from pointer import PhaseLinkageComp
from pointer.phases import GaussLobattoPhase, RadauPseudospectralPhase

from pointer.examples.min_time_climb_2d.min_time_climb_ode import MinTimeClimbODE

_phase_map = {'gauss-lobatto': GaussLobattoPhase,
              'radau-ps': RadauPseudospectralPhase}


def min_time_climb_problem(optimizer='SLSQP', num_seg=3, transcription_order=5,
                           transcription='gauss-lobatto', alpha_guess=False,
                           top_level_densejacobian=True, simul_derivs=False,
                           mbi_thrust=False):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    # p.driver.options['simul_derivs'] = simul_derivs
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['Iterations limit'] = 100000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
        p.driver.opt_settings['Verify level'] = 3
        p.driver.opt_settings['Function precision'] = 1.0E-6
        p.driver.opt_settings['Linesearch tolerance'] = 0.10
        p.driver.opt_settings['Major step limit'] = 0.5

    phase_class = _phase_map[transcription]

    climb = phase_class(ode_function=MinTimeClimbODE(mbi_thrust=mbi_thrust),
                        num_segments=num_seg,
                        transcription_order=transcription_order,
                        compressed=False)

    climb.set_time_options(opt_initial=False, duration_bounds=(50, None),
                           duration_ref=100.0)

    climb.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

    climb.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    climb.set_state_options('v', fix_initial=True, lower=10.0,
                            scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

    climb.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    climb.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                            scaler=1.0E-3, defect_scaler=1.0E-3)

    climb.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      dynamic=True, rate_continuity=True)

    climb.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    climb.add_control('Isp', val=1600.0, units='s', dynamic=False, opt=False)

    climb.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
    climb.add_boundary_constraint('aero.mach', loc='final', equals=1.0, units=None)
    climb.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

    climb.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    climb.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)


    # Set up the loiter phase
    loiter = phase_class(ode_function=MinTimeClimbODE(mbi_thrust=mbi_thrust),
                        num_segments=num_seg,
                        transcription_order=transcription_order,
                        compressed=False)

    loiter.set_time_options(opt_initial=False, duration_bounds=(50, 400),
                           duration_ref=100.0)

    loiter.set_state_options('r', lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

    loiter.set_state_options('h', lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    loiter.set_state_options('v', lower=10.0,
                            scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

    loiter.set_state_options('gam', lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    loiter.set_state_options('m', lower=10.0, upper=1.0E5,
                            scaler=1.0E-3, defect_scaler=1.0E-3)

    loiter.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      dynamic=True, rate_continuity=True)

    loiter.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    loiter.add_control('Isp', val=1600.0, units='s', dynamic=False, opt=False)

    loiter.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    loiter.add_path_constraint(name='aero.mach', lower=0.1, upper=1.)

    p.model.add_subsystem('climb', climb)
    p.model.add_subsystem('loiter', loiter)

    # Connect the phases
    linkage_comp = PhaseLinkageComp()
    linkage_comp.add_linkage(name='L01', vars=['t'], units='s', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['r', 'h'], units='m', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['v'], units='m/s', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['gam'], units='rad', equals=0.0, linear=True)
    linkage_comp.add_linkage(name='L01', vars=['m'], units='kg', equals=0.0, linear=True)

    # p.model.connect('climb.time++', 'linkages.L01_t:lhs')
    # p.model.connect('loiter.time--', 'linkages.L01_t:rhs')

    p.model.connect('climb.states:r++', 'linkages.L01_r:lhs')
    p.model.connect('loiter.states:r--', 'linkages.L01_r:rhs')

    p.model.connect('climb.states:h++', 'linkages.L01_h:lhs')
    p.model.connect('loiter.states:h--', 'linkages.L01_h:rhs')

    p.model.connect('climb.states:v++', 'linkages.L01_v:lhs')
    p.model.connect('loiter.states:v--', 'linkages.L01_v:rhs')

    p.model.connect('climb.states:gam++', 'linkages.L01_gam:lhs')
    p.model.connect('loiter.states:gam--', 'linkages.L01_gam:rhs')

    p.model.connect('climb.states:m++', 'linkages.L01_m:lhs')
    p.model.connect('loiter.states:m--', 'linkages.L01_m:rhs')

    p.model.add_subsystem('linkages', linkage_comp)

    if top_level_densejacobian:
        p.model.jacobian = CSCJacobian()
        p.model.linear_solver = DirectSolver()

    p.model.add_constraint('climb.tp', upper=325.)

    distance = 4e5

    # Minimize time at the end of the phase
    loiter.set_objective('time', loc='final', ref=100.0)
    loiter.add_boundary_constraint('r', loc='final', equals=distance)

    p.setup(mode='fwd')

    p['climb.t0'] = 0.0
    p['climb.tp'] = 320.
    p['climb.states:r'] = climb.interpolate(ys=[0.0, 111319.54], nodes='disc')
    p['climb.states:h'] = climb.interpolate(ys=[100.0, 20000.0], nodes='disc')
    p['climb.states:v'] = climb.interpolate(ys=[135.964, 283.159], nodes='disc')
    p['climb.states:gam'] = climb.interpolate(ys=[0.0, 0.0], nodes='disc')
    p['climb.states:m'] = climb.interpolate(ys=[19030.468, 16841.431], nodes='disc')
    p['climb.controls:alpha'] = climb.interpolate(ys=[0.0, 0.0], nodes='all')

    if alpha_guess:
        interp_times = [0.0, 9.949, 19.9, 29.85, 39.8, 49.74, 59.69, 69.64, 79.59, 89.54, 99.49,
                        109.4, 119.4, 129.3, 139.3, 149.2, 159.2, 169.1, 179.1, 189.0, 199.0,
                        208.9, 218.9, 228.8, 238.8, 248.7, 258.7, 268.6, 278.6, 288.5, 298.5]

        interp_alphas = [0.07423, 0.03379, 0.01555, 0.01441, 0.02529, 0.0349, 0.02994, 0.02161,
                         0.02112, 0.02553, 0.0319, 0.03622, 0.03443, 0.03006, 0.0266, 0.0244,
                         0.02384, 0.02399, 0.02397, 0.02309, 0.0207, 0.01948, 0.0221, 0.02929,
                         0.04176, 0.05589, 0.06805, 0.06952, 0.05159, -0.002399, -0.1091]

        p['climb.controls:alpha'] = climb.interpolate(xs=interp_times,
                                                       ys=interp_alphas, nodes='all')

    p['loiter.t0'] = 320.
    p['loiter.tp'] = 1000.
    p['loiter.states:r'] = loiter.interpolate(ys=[110000., distance], nodes='disc')
    p['loiter.states:h'] = loiter.interpolate(ys=[20000.0, 20000.0], nodes='disc')
    p['loiter.states:v'] = loiter.interpolate(ys=[283.159, 283.159], nodes='disc')
    p['loiter.states:gam'] = loiter.interpolate(ys=[0.0, 0.0], nodes='disc')
    p['loiter.states:m'] = loiter.interpolate(ys=[16000., 1000.], nodes='disc')
    p['loiter.controls:alpha'] = loiter.interpolate(ys=[0.0, 0.0], nodes='all')



    return p


if __name__ == '__main__':
    p = min_time_climb_problem(optimizer='SNOPT', num_seg=8, transcription_order=3)
    from openmdao.api import view_model
    # view_model(p)
    p.run_model()
    p.run_driver()

    import matplotlib.pyplot as plt

    h = p['climb.states:h']
    r = p['climb.states:r']
    mach = p['climb.rhs_disc.aero.mach_comp.mach']

    h_loiter = p['loiter.states:h']
    r_loiter = p['loiter.states:r']
    mach_loiter = p['loiter.rhs_disc.aero.mach_comp.mach']

    ax1 = plt.subplot(211)
    plt.plot(r, h, color='C0')
    plt.plot(r_loiter, h_loiter, color='C1')

    ax2 = plt.subplot(212)
    plt.plot(r, mach, color='C0')
    plt.plot(r_loiter, mach_loiter, color='C1')

    plt.show()
