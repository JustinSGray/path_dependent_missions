from __future__ import print_function, division, absolute_import
import numpy as np
import sys
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from openmdao.api import Problem, Group, IndepVarComp
from dymos.models.atmosphere import StandardAtmosphereGroup

from path_dependent_missions.escort.aero import AeroGroup

from niceplots import parula, draggable_legend


cmap = parula.parula_map

def setup_prob(nn, h, v):
    p = Problem()
    p.model = Group()

    ivc = IndepVarComp()
    ivc.add_output('h', shape=nn, val=h, units='ft')
    ivc.add_output('v', shape=nn, val=v, units='m/s')
    ivc.add_output('alpha', shape=nn, val=np.linspace(-5., 10., nn), units='deg')
    ivc.add_output('S', shape=nn, val=49.236, units='m**2')

    p.model.add_subsystem('ivc', ivc, promotes=['*'])

    p.model.add_subsystem(name='atmos',
                       subsys=StandardAtmosphereGroup(num_nodes=nn),
                       promotes=['*'])

    p.model.add_subsystem(name='aero',
                       subsys=AeroGroup(num_nodes=nn),
                       promotes=['*'])

    p.setup()

    return p

def plot_drag_polar():

    fig, axarr = plt.subplots(1, 3, figsize=(15, 5))

    plt.sca(axarr[0])

    v_list = np.linspace(0., 606.5, 100)
    prob = setup_prob(50, 30000., v_list[0])

    alpha = prob['alpha']

    for i, v in enumerate(v_list):
        prob['v'][:] = v
        prob.run_model()
        CL = prob['CL']
        CD = prob['CD']
        axarr[0].plot(CD, CL, color=cmap(i/len(v_list)), label=str(round(prob['mach'][0], 2)))
        axarr[1].plot(alpha, CL, color=cmap(i/len(v_list)), label=str(round(prob['mach'][0], 2)))
        axarr[2].plot(alpha, CD, color=cmap(i/len(v_list)), label=str(round(prob['mach'][0], 2)))

    axarr[0].set_xlabel('CD')
    axarr[0].set_ylabel('CL')

    axarr[1].set_xlabel('alpha')
    axarr[1].set_ylabel('CL')

    axarr[2].set_xlabel('alpha')
    axarr[2].set_ylabel('CD')

    axarr[0].annotate('', xy=(.5, .5), xytext=(.15, .82), xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->, head_width=.25', facecolor='gray'),
            )
    axarr[0].annotate('increasing Mach', xy=(.5, .5), xytext=(.04, .85), xycoords='axes fraction',
            rotation=0.)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_drag_polar()
