from __future__ import print_function, division, absolute_import
import matplotlib
# matplotlib.use('agg')
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from dymos import Phase

from path_dependent_missions.thermal_mission.thermal_mission_problem import thermal_mission_problem
from path_dependent_missions.utils.gen_mission_plot import save_results, plot_results

options = {
    'transcription' : 'gauss-lobatto',
    'num_seg' : 25,
    'transcription_order' : 3,
    'm_recirculated' : 20.,
    'opt_m_recirculated' : True,
    'Q_sink' : 0.e3,
    'Q_out' : 5.e3,
    'T' : 312.,
    'm_initial' : 20.e3,
    'opt_throttle' : True,
    'opt_m' : True,
    'engine_heat_coeff' : 0.,
    }

pump_heat_coeff_list = np.linspace(2., 4., 3)*1e4

# for i, pump_heat_coeff in enumerate(pump_heat_coeff_list):
#     options['pump_heat_coeff'] = pump_heat_coeff
#     p = thermal_mission_problem(**options)
#     p.run_driver()
#     save_results(p, 'pump_{}.pkl'.format(i), options)

plot_list = ['pump_{}.pkl'.format(i) for i in range(len(pump_heat_coeff_list))]
f, axarr = plot_results(plot_list, save_fig=False, figsize=(12, 12), color_offset=4)

# axarr[0].annotate('', xy=(.75, .25), xytext=(.1, .75), xycoords='axes fraction',
#         arrowprops=dict(arrowstyle='->, head_width=.25', facecolor='gray'))
axarr[0].annotate('increasing $q_{pump}$', xy=(.5, .82), xytext=(.05, .77), xycoords='axes fraction',
        arrowprops=dict(arrowstyle='->, head_width=.25', facecolor='gray'), rotation=0.)

import matplotlib.pyplot as plt
# plt.show()
plt.savefig('pump_compare.pdf')
