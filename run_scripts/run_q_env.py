from __future__ import print_function, division, absolute_import
import matplotlib
# matplotlib.use('agg')
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
from dymos import Phase

from path_dependent_missions.thermal_mission.thermal_mission_problem import thermal_mission_problem
from path_dependent_missions.utils.gen_mission_plot import save_results, plot_results

options = {
    'transcription' : 'gauss-lobatto',
    'num_seg' : 25,
    'transcription_order' : 3,
    'm_recirculated' : 0.,
    'opt_m_recirculated' : False,
    'Q_sink' : 100.e3,
    'Q_out' : 5.e3,
    'T' : 312.,
    'm_initial' : 20.e3,
    'opt_throttle' : True,
    'opt_m' : True,
    'engine_heat_coeff' : 0.,
    }

Q_env_list = np.linspace(4., 5., 3)*1e5

# for i, Q_env in enumerate(Q_env_list):
#     options['Q_env'] = Q_env
#     p = thermal_mission_problem(**options)
#     p.run_driver()
#     save_results(p, 'Q_env_{}.pkl'.format(i), options)

plot_list = ['Q_env_{}.pkl'.format(i) for i in range(len(Q_env_list))]
f, axarr = plot_results(plot_list, save_fig=False, figsize=(12, 12))

axarr[0].annotate('', xy=(.75, .25), xytext=(.55, .75), xycoords='axes fraction',
        arrowprops=dict(arrowstyle='->, head_width=.25', facecolor='gray'))
axarr[0].annotate('increasing $\dot Q_{env}$', xy=(.5, .5), xytext=(.44, .82), xycoords='axes fraction', rotation=0.)

import matplotlib.pyplot as plt
# plt.show()
plt.savefig('Q_env_comparisons.pdf')
