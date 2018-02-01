from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from pointer.phases import GaussLobattoPhase, RadauPseudospectralPhase

from path_dependent_missions.simple_heat.simple_heat_ode import SimpleHeatODE
import numpy as np


def setup_energy_opt(num_seg, order, q_tank, q_hx1, q_hx2, opt_burn=False):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['Major iterations limit'] = 2000
    p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-7
    p.driver.opt_settings['Major optimality tolerance'] = 1.0E-7
    p.driver.opt_settings['Verify level'] = -1

    # phase = RadauPseudospectralPhase(ode_function=SimpleHeatODE(), num_segments=num_seg, transcription_order=order, compressed=False)
    phase = GaussLobattoPhase(ode_function=SimpleHeatODE(q_tank=q_tank, q_hx1=q_hx1, q_hx2=q_hx2), num_segments=num_seg, transcription_order=order, compressed=False)

    phase.set_time_options(opt_initial=False, opt_duration=False)

    phase.set_state_options('m', lower=1., upper=10., fix_initial=True)
    phase.set_state_options('T', fix_initial=True, defect_scaler=.01)
    phase.set_state_options('energy', fix_initial=True)
    phase.set_objective('energy', loc='final')

    phase.add_control('m_flow', opt=True, lower=0., upper=5., dynamic=True)
    if opt_burn:
        phase.add_control('m_burn', opt=opt_burn, lower=.2, upper=5., dynamic=False)
    else:
        phase.add_control('m_burn', opt=opt_burn, dynamic=True)
    phase.add_path_constraint('T', upper=1.)
    phase.add_path_constraint('m_flow_rate', upper=0.)
    phase.add_path_constraint('fuel_burner.m_recirculated', lower=0.)

    p.model.add_subsystem('phase', phase)

    p.setup(check=True, force_alloc_complex=True)

    p['phase.states:m'] = 10.
    p['phase.states:T'] = 1.
    p['phase.states:energy'] = 0.
    p['phase.controls:m_flow'] = .5
    p['phase.controls:m_burn'] = .1
    p['phase.t_initial'] = 0.
    p['phase.t_duration'] = 1.

    return p

def plot_results(p):
    m = p.model.phase.get_values('m', nodes='all')
    time = p.model.phase.get_values('time', nodes='all')
    T = p.model.phase.get_values('T', nodes='all')
    m_flow = p.model.phase.get_values('m_flow', nodes='all')
    energy = p.model.phase.get_values('energy', nodes='all')

    out = p.model.phase.simulate(times=np.linspace(0, 1, 100))

    m2 = out.get_values('m')
    time2 = out.get_values('time')
    T2 = out.get_values('T')
    m_flow2 = out.get_values('m_flow')
    energy2 = out.get_values('energy')

    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(4, sharex=True)
    axarr[0].scatter(time, m)
    axarr[0].plot(time2, m2)
    axarr[0].set_xlabel('time')
    axarr[0].set_ylabel('mass')

    axarr[1].scatter(time, T)
    axarr[1].plot(time2, T2)
    axarr[1].set_xlabel('time')
    axarr[1].set_ylabel('temp')

    axarr[2].scatter(time, m_flow)
    axarr[2].plot(time2, m_flow2)
    axarr[2].set_xlabel('time')
    axarr[2].set_ylabel('m_flow')

    axarr[3].scatter(time, energy)
    axarr[3].plot(time2, energy2)
    axarr[3].set_xlabel('time')
    axarr[3].set_ylabel('energy')

    plt.show()
