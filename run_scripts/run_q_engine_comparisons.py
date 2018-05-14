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
    'num_seg' : 12,
    'transcription_order' : 3,
    'm_recirculated' : 0.,
    'opt_m_recirculated' : False,
    'Q_env' : 100.e3,
    'Q_sink' : 0.e3,
    'Q_out' : 0.e3,
    'T' : 315.,
    # 'T_o' : 325,
    'm_initial' : 40.e3,
    'opt_throttle' : True,
    'opt_m' : True,
    }

coeff_list = np.linspace(6., 9., 3)*1e5

for i, engine_heat_coeff in enumerate(coeff_list):
    options['engine_heat_coeff'] = engine_heat_coeff
    p = thermal_mission_problem(**options)
    p.run_driver()
    save_results(p, 'engine_heat_coeff_{}.pkl'.format(i), options)

plot_list = ['engine_heat_coeff_{}.pkl'.format(i) for i in range(len(coeff_list))]
f, axarr = plot_results(plot_list, save_fig=True, list_to_plot=['h', 'aero.mach', 'm_fuel', 'T', 'm_burn', 'throttle'], figsize=(8, 8))

axarr[0].annotate('', xy=(.75, .25), xytext=(.55, .75), xycoords='axes fraction',
        arrowprops=dict(arrowstyle='->, head_width=.25', facecolor='gray'))
axarr[0].annotate('increasing Q_engine', xy=(.5, .5), xytext=(.44, .82), xycoords='axes fraction', rotation=0.)

import matplotlib.pyplot as plt
# plt.show()
plt.savefig('engine_coeff_compare.pdf')
