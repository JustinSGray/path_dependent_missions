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
from path_dependent_missions.escort.aero.aero_smt import AeroSMTGroup

from niceplots import parula, draggable_legend


cmap = parula.parula_map

def setup_prob(nn, h, v):
    p = Problem()
    p.model = Group()

    ivc = IndepVarComp()
    ivc.add_output('h', shape=nn, val=h, units='ft')
    ivc.add_output('v', shape=nn, val=v, units='m/s')
    ivc.add_output('alpha', shape=nn, val=np.linspace(-8., 8., nn), units='deg')
    ivc.add_output('S', shape=nn, val=49.236, units='m**2')

    p.model.add_subsystem('ivc', ivc, promotes=['*'])

    p.model.add_subsystem(name='atmos',
                       subsys=StandardAtmosphereGroup(num_nodes=nn),
                       promotes=['*'])

    p.model.add_subsystem(name='aero',
                       subsys=AeroGroup(num_nodes=nn),
                       promotes_inputs=['*'])

    p.model.add_subsystem(name='aero_smt',
                        subsys=AeroSMTGroup(num_nodes=nn),
                        promotes_inputs=['*'])

    p.setup()

    return p

def plot_drag_polar():

    fig, axarr = plt.subplots(1, 3, figsize=(15, 5))

    plt.sca(axarr[0])

    v_list = np.linspace(0., 550., 100)
    prob = setup_prob(50, 30000., v_list[0])

    alpha = prob['alpha']

    for i, v in enumerate(v_list):
        prob['v'][:] = v
        prob.run_model()
        CL_ = prob['aero.CL']
        CD_ = prob['aero.CD']

        CL = prob['aero_smt.CL'] / 49.236/2.
        CD = prob['aero_smt.CD'] / 49.236/2.

        axarr[0].plot(CD, CL, color=cmap(i/len(v_list)), label=str(round(prob['aero.mach'][0], 2)))
        axarr[1].plot(alpha, CL, color=cmap(i/len(v_list)), label=str(round(prob['aero.mach'][0], 2)))
        axarr[2].plot(alpha, CD, color=cmap(i/len(v_list)), label=str(round(prob['aero.mach'][0], 2)))

    axarr[0].set_xlabel('CD')
    axarr[0].set_ylabel('CL')

    axarr[1].set_xlabel('alpha')
    axarr[1].set_ylabel('CL')

    axarr[2].set_xlabel('alpha')
    axarr[2].set_ylabel('CD')

    axarr[0].annotate('', xy=(.5, .5), xytext=(.15, .82), xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->, head_width=.25', facecolor='gray'),
            )
    axarr[0].annotate('increasing Mach', xy=(.5, .5), xytext=(.04, .88), xycoords='axes fraction',
            rotation=0.)

    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0, vmax=1.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.tight_layout()

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    clb = fig.colorbar(sm, ticks=np.linspace(0, 1.8, 4),
                 boundaries=np.arange(0., 1.81, .01), cax=cbar_ax)
    cbar_ax.set_title('Mach', pad=12)

    # plt.show()
    plt.savefig('polars.pdf')

if __name__ == "__main__":
    plot_drag_polar()
