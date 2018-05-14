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
    'm_recirculated' : 10.,
    'opt_m_recirculated' : True,
    'Q_env' : 100.e3,
    'Q_sink' : 100.e3,
    'Q_out' : 5.e3,
    'engine_heat_coeff' : 900.e3,
    'T' : 315.,
    # 'T_o' : 325,
    'm_initial' : 21.e3,
    'opt_throttle' : True,
    'opt_m' : True,
    }

# p = thermal_mission_problem(**options)
# p.run_driver()
# save_results(p, 'recirc.pkl', options)
#
# options = {
#     'transcription' : 'gauss-lobatto',
#     'num_seg' : 12,
#     'transcription_order' : 3,
#     'm_recirculated' : 0.,
#     'opt_m_recirculated' : False,
#     'Q_env' : 100.e3,
#     'Q_sink' : 100.e3,
#     'Q_out' : 5.e3,
#     'engine_heat_coeff' : 900.e3,
#     'T' : 315.,
#     # 'T_o' : 325,
#     'm_initial' : 21.e3,
#     'opt_throttle' : True,
#     'opt_m' : True,
#     }
#
# p = thermal_mission_problem(**options)
# p.run_driver()
# save_results(p, 'no_recirc.pkl', options)
#
plot_list = ['no_recirc.pkl', 'recirc.pkl']
f, axarr = plot_results(plot_list, save_fig=True, list_to_plot=['h', 'aero.mach', 'm_fuel', 'T', 'm_burn', 'm_recirculated', 'throttle'], figsize=(8, 8))

import matplotlib.pyplot as plt
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

axarr[0].annotate('no recirculation', xy=(.5, .5), xytext=(.8, .3), xycoords='axes fraction', rotation=0., color=colors[0])

axarr[0].annotate('recirculation', xy=(.5, .5), xytext=(.5, .6), xycoords='axes fraction', rotation=0., color=colors[1])


# plt.show()
plt.savefig('recirc.pdf')