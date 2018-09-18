from __future__ import print_function, division, absolute_import
import matplotlib
# matplotlib.use('agg')
import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver

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
    'Q_sink' : 100.e3,
    'Q_out' : 5.e3,
    'T' : 312.,
    'm_initial' : 20.e3,
    'opt_throttle' : True,
    'opt_m' : True,
    }

# p = thermal_mission_problem(**options)
# p.run_driver()
# save_results(p, 'recirc.pkl', options)
#
# options['opt_m_recirculated'] = False
# options['m_recirculated'] = 0.
#
# p = thermal_mission_problem(**options)
# p.run_driver()
# save_results(p, 'no_recirc.pkl', options)

plot_list = ['no_recirc.pkl', 'recirc.pkl']
color_offset = 2
f, axarr = plot_results(plot_list, save_fig=False, figsize=(12, 12), color_offset=color_offset)

import matplotlib.pyplot as plt
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

axarr[0].annotate('no recirculation', xy=(.5, .5), xytext=(.75, .4), xycoords='axes fraction', rotation=0., color=colors[0+color_offset])

axarr[0].annotate('recirculation', xy=(.5, .5), xytext=(.45, .6), xycoords='axes fraction', rotation=0., color=colors[1+color_offset])


# plt.show()
plt.savefig('recirc.pdf')
