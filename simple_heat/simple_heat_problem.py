from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from pointer.phases import GaussLobattoPhase, RadauPseudospectralPhase

from path_dependent_missions.simple_heat.simple_heat_ode import SimpleHeatODE


p = Problem(model=Group())

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.opt_settings['Major iterations limit'] = 2000
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
p.driver.opt_settings['Verify level'] = -1

phase = RadauPseudospectralPhase(ode_function=SimpleHeatODE(), num_segments=2, transcription_order=3, compressed=False)

phase.set_time_options(opt_initial=False, opt_duration=False)

phase.set_state_options('m', lower=1., upper=10., fix_initial=True)
phase.set_state_options('T', fix_initial=True)
phase.set_objective('power', loc='final')

phase.add_control('m_flow', opt=True, val=1., lower=.2, dynamic=True)
phase.add_control('m_burn', opt=False, val=0., dynamic=False)
phase.add_path_constraint('T', upper=1.)
phase.add_path_constraint('fuel_burner.m_recirculated', lower=0.)
phase.add_path_constraint('m', lower=1.)

p.model.add_subsystem('phase', phase)

p.setup(check=True)

p['phase.states:m'] = phase.interpolate(ys=[10., 10.], nodes='disc')
p['phase.states:T'] = phase.interpolate(ys=[0.5, 0.], nodes='disc')
p['phase.controls:m_flow'] = phase.interpolate(ys=[3., 3.], nodes='disc')

if 0:
    p.run_model()
    out = p.model.phase.simulate()
    p.check_partials(compact_print=True)

    # from openmdao.api import view_model
    # view_model(p)

    print(out.keys())
    m = out['states:m']
    time = out['time']
    T = out['states:T']
    m_flow = out['controls:m_flow']


else:
    p.run_driver()

    m = p.model.phase.get_values('m', nodes='all')
    time = p.model.phase.get_values('time', nodes='all')
    T = p.model.phase.get_values('T', nodes='all')
    m_flow = p.model.phase.get_values('m_flow', nodes='all')

print(m)

if 1:

    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(time, m)
    axarr[0].set_xlabel('time')
    axarr[0].set_ylabel('mass')

    axarr[1].plot(time, T)
    axarr[1].set_xlabel('time')
    axarr[1].set_ylabel('temp')

    axarr[2].plot(time, m_flow)
    axarr[2].set_xlabel('time')
    axarr[2].set_ylabel('m_flow')

    plt.show()
