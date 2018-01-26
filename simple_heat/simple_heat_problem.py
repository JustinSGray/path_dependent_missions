from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from pointer.phases import GaussLobattoPhase, RadauPseudospectralPhase

from path_dependent_missions.simple_heat.simple_heat_ode import SimpleHeatODE

_phase_map = {'gauss-lobatto': GaussLobattoPhase,
              'radau-ps': RadauPseudospectralPhase}


def simple_heat_problem(optimizer='SNOPT', num_seg=3, transcription_order=5,
                           transcription='gauss-lobatto'):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
        p.driver.opt_settings['Verify level'] = -1
        p.driver.opt_settings['Function precision'] = 1.0E-6
        p.driver.opt_settings['Linesearch tolerance'] = 0.10
        p.driver.opt_settings['Major step limit'] = 0.5

    phase_class = _phase_map[transcription]

    phase = phase_class(ode_function=SimpleHeatODE(),
                        num_segments=num_seg,
                        transcription_order=transcription_order,
                        compressed=False)

    phase.set_time_options(opt_initial=False, opt_duration=True, duration_bounds=(0, 2))

    phase.set_state_options('m', lower=0., upper=10., fix_initial=True)

    phase.set_objective('time', loc='final', scaler=-1)
    phase.add_control('m_flow', opt=True, val=1., dynamic=False)
    phase.add_boundary_constraint('m', loc='final', equals=0.)

    p.model.add_subsystem('phase', phase)

    p.setup()

    p['phase.states:m'] = phase.interpolate(ys=[10, 10], nodes='disc')

    return p


if __name__ == '__main__':
    p = simple_heat_problem(optimizer='SNOPT', num_seg=4, transcription_order=1, transcription='radau-ps')

    # p.run_model()
    # p.model.phase.simulate()

    p.run_driver()

    m = p.model.phase.get_values('m', nodes='all')
    print(m)

    # from openmdao.api import view_model
    # view_model(p)
    # exit()
