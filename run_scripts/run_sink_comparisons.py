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
    'm_recirculated' : 50.,
    'opt_m_recirculated' : True,
    'Q_env' : 500.e3,
    'Q_sink' : 0.e3,
    'Q_out' : 5.e3,
    'engine_heat_coeff' : 0.e3,
    'T' : 312.,
    # 'T_o' : 325,
    'm_initial' : 21.e3,
    'opt_throttle' : True,
    'opt_m' : True,
    }

p = thermal_mission_problem(**options)
p.run_driver()
save_results(p, 'no_sink.pkl', options)

options['Q_sink'] = 200.e3
options['Q_env'] = 300.e3

p = thermal_mission_problem(**options)
p.run_driver()
save_results(p, 'sink.pkl', options)

plot_list = ['no_sink.pkl', 'sink.pkl']
f, axarr = plot_results(plot_list, save_fig=True, figsize=(12, 12))

import matplotlib.pyplot as plt
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

axarr[0].annotate('no_sink', xy=(.5, .5), xytext=(.65, .3), xycoords='axes fraction', rotation=0., color=colors[0])

axarr[0].annotate('sink', xy=(.5, .5), xytext=(.5, .6), xycoords='axes fraction', rotation=0., color=colors[1])


plt.show()
# plt.savefig('sink.pdf')
