from __future__ import print_function, division, absolute_import
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian, SqliteRecorder

from pointer.phases import GaussLobattoPhase, RadauPseudospectralPhase

from path_dependent_missions.escort.min_time_climb_ode import MinTimeClimbODE

_phase_map = {'gauss_lobatto': GaussLobattoPhase,
              'radau_ps': RadauPseudospectralPhase}

optimizer = 'SNOPT'
num_seg = 3
transcription_order = 3
transcription = 'gauss_lobatto'


p = Problem(model=Group())

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = optimizer
if optimizer == 'SNOPT':
    p.driver.opt_settings['Major iterations limit'] = 500
    p.driver.opt_settings['iSumm'] = 6
    p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-9
    p.driver.opt_settings['Major optimality tolerance'] = 1.0E-9
    p.driver.opt_settings['Verify level'] = 3

phase_class = _phase_map[transcription]

phase = phase_class(ode_function=MinTimeClimbODE(),
                    num_segments=num_seg,
                    transcription_order=transcription_order,
                    compressed=False)

p.model.add_subsystem('phase', phase)

phase.set_time_options(opt_initial=False, duration_bounds=(50, 1e5),
                       duration_ref=100.0)

phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                        scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

phase.set_state_options('h', fix_initial=True, lower=0, upper=14000.0,
                        scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

phase.set_state_options('v', fix_initial=True, lower=0.01,
                        scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                        ref=1.0, defect_scaler=1.0, units='rad')

phase.set_state_options('m', fix_initial=True, lower=1e3, upper=1.0E6,
                        scaler=1.0E-3, defect_scaler=1.0E-3)

phase.add_control('alpha', units='rad', lower=-8. * np.pi/180., upper=8. * np.pi/180., scaler=1,
                  dynamic=True, rate_continuity=True)

phase.add_control('throttle', val=1.0, lower=0., upper=1., dynamic=True, opt=True)

phase.add_boundary_constraint('h', loc='final', equals=1e3, scaler=1.0E-3, units='m')
phase.add_boundary_constraint('r', loc='final', equals=15000, units=None)
# phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

phase.add_path_constraint(name='aero.mach', lower=0.01, upper=.8)
phase.add_path_constraint(name='prop.m_dot', upper=0.)
phase.add_path_constraint(name='flight_dynamics.r_dot', lower=0.)
# phase.add_path_constraint(name='m', lower=1e4)
# phase.add_path_constraint(name='h', lower=0.)

phase.set_objective('time', loc='final', ref=10.0)
# phase.set_objective('m', loc='final', ref=-100.0)
# phase.set_objective('r', loc='final', ref=-100000.0)

p.model.jacobian = CSCJacobian()
p.model.linear_solver = DirectSolver()

# p.driver.add_recorder(SqliteRecorder('out.db'))
p.setup(mode='fwd', check=True)
# from openmdao.api import view_model
# view_model(p)
# exit()

p['phase.t_initial'] = 0.0
p['phase.t_duration'] = 50.

p['phase.states:r'] = phase.interpolate(ys=[0.0, 12000], nodes='disc')
p['phase.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='disc')
p['phase.states:m'] = phase.interpolate(ys=[2e4, 1.9e4], nodes='disc')
p['phase.controls:alpha'] = phase.interpolate(ys=[0., 0.], nodes='all')

p['phase.states:h'][:] = 1e3
p['phase.states:v'][:] = 267.

# Create CRM geometry
# for phase_name in ['phase.rhs_disc.aero.OAS_group.', 'phase.rhs_col.aero.OAS_group.']:
#     p[phase_name + 'wing_chord_dv'] = np.array([ 107.4 , 285.8 , 536.2 , 285.8 , 107.4 ]) * 0.0254
#     p[phase_name + 'wing_twist_dv'] = np.array([ -3.75 ,  0.76 ,  6.72 ,  0.76 , -3.75 ]) * np.pi / 180.
#     p[phase_name + 'wing_sec_x_dv'] = np.array([  1780 ,  1226 ,   904 ,  1226 ,  1780 ]) * 0.0254
#     p[phase_name + 'wing_sec_y_dv'] = np.array([ 263.8 , 181.1 , 174.1 , 181.1 , 263.8 ]) * 0.0254
#     p[phase_name + 'wing_sec_z_dv'] = np.array([ -1157 ,  -428 ,     0 ,   428 ,  1157 ]) * 0.0254

p.run_model()

exp_out = p.model.phase.simulate(times='disc')
p['phase.states:r'] = np.atleast_2d(exp_out.get_values('r')).T
p['phase.states:h'] = np.atleast_2d(exp_out.get_values('h')).T
p['phase.states:v'] = np.atleast_2d(exp_out.get_values('v')).T
p['phase.states:gam'] = np.atleast_2d(exp_out.get_values('gam')).T
p['phase.states:m'] = np.atleast_2d(exp_out.get_values('m')).T

p.run_driver()



# Plotting below here
import numpy as np
phase = p.model.phase
import matplotlib.pyplot as plt

time = p.model.phase.get_values('time', nodes='all')
m = p.model.phase.get_values('m', nodes='all')
mach = p.model.phase.get_values('aero.mach', nodes='all')
h = p.model.phase.get_values('h', nodes='all')
r = p.model.phase.get_values('r', nodes='all')
alpha = p.model.phase.get_values('alpha', nodes='all')

f, axarr = plt.subplots(4, sharex=True)

axarr[0].plot(r, h, 'ko')
axarr[0].set_xlabel('range')
axarr[0].set_ylabel('altitude')

axarr[1].plot(r, mach, 'ko')
axarr[1].set_xlabel('range')
axarr[1].set_ylabel('mach')

axarr[2].plot(r, m, 'ko')
axarr[2].set_xlabel('range')
axarr[2].set_ylabel('mass')

axarr[3].plot(r, alpha, 'ko')
axarr[3].set_xlabel('range')
axarr[3].set_ylabel('alpha')

if 1:
    exp_out = phase.simulate(times=np.linspace(0, p['phase.t_duration'], 20))
    time2 = exp_out.get_values('time')
    m2 = exp_out.get_values('m')
    mach2 = exp_out.get_values('aero.mach')
    h2 = exp_out.get_values('h')
    r2 = exp_out.get_values('r')
    alpha2 = exp_out.get_values('alpha')
    axarr[0].plot(r2, h2)
    axarr[1].plot(r2, mach2)
    axarr[2].plot(r2, m2)
    axarr[3].plot(r2, alpha2)

plt.show()
