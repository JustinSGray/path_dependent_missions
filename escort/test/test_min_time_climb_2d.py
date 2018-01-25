from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from path_dependent_missions.escort.min_time_climb_problem import min_time_climb_problem

SHOW_PLOTS = False


class TestMinTimeClimb2D(unittest.TestCase):

    def test_min_time_climb_gl(self):
        p = min_time_climb_problem(optimizer='SLSQP', num_seg=10,
                                   transcription_order=[5, 3, 5, 3, 5, 3, 5, 3, 5, 3],
                                   transcription='gauss-lobatto', top_level_densejacobian=True)

        p.run_driver()

        phase0 = p.model.phase0

        # Check that time matches to within 1% of an externally verified solution.
        assert_almost_equal((phase0.get_values('time')[-1]-320.0)/320.0, 0.0, decimal=2)

        exp_out = phase0.simulate(times=np.linspace(0, p['phase0.tp'], 100))
        if SHOW_PLOTS:

            import matplotlib.pyplot as plt
            plt.plot(phase0.get_values('time'), phase0.get_values('h'), 'ro')
            plt.plot(exp_out['time'], exp_out['states:h'], 'b-')
            plt.xlabel('time (s)')
            plt.ylabel('altitude (m)')

            plt.figure()
            plt.plot(phase0.get_values('v'), phase0.get_values('h'), 'ro')
            plt.plot(exp_out['states:v'], exp_out['states:h'], 'b-')
            plt.xlabel('airspeed (m/s)')
            plt.ylabel('altitude (m)')

            plt.show()

    def test_min_time_climb_radau(self):
        p = min_time_climb_problem(optimizer='SLSQP', num_seg=10, transcription_order=3,
                                   transcription='radau-ps')

        p.run_driver()

        phase0 = p.model.phase0

        # Check that time matches to within 1% of an externally verified solution.
        assert_almost_equal((phase0.get_values('time')[-1]-320.0)/320.0, 0.0, decimal=2)

        exp_out = phase0.simulate(times=np.linspace(0, p['phase0.tp'], 100))

        if SHOW_PLOTS:
            import matplotlib.pyplot as plt
            plt.plot(phase0.get_values('time'), phase0.get_values('h'), 'ro')
            plt.plot(exp_out['time'], exp_out['states:h'], 'b-')
            plt.xlabel('time (s)')
            plt.ylabel('altitude (m)')

            plt.figure()
            plt.plot(phase0.get_values('v'), phase0.get_values('h'), 'ro')
            plt.plot(exp_out['states:v'], exp_out['states:h'], 'b-')
            plt.xlabel('airspeed (m/s)')
            plt.ylabel('altitude (m)')

            plt.show()


if __name__ == '__main__':
    unittest.main()
