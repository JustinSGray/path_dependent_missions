from __future__ import print_function

import sys
import numpy as np

from openmdao.api import ImplicitComponent, Group, IndepVarComp, BalanceComp, DirectSolver, BoundsEnforceLS, NewtonSolver, ArmijoGoldsteinLS
from openmdao.core.component import Component

from pycycle.constants import AIR_MIX, AIR_FUEL_MIX
from pycycle.connect_flow import connect_flow
# from pycycle.balance import Balance
from pycycle.cea import species_data
from pycycle.elements.api import FlightConditions, Inlet, Compressor, Combustor, Turbine, Nozzle, Shaft, Duct, Performance,Splitter, Mixer, BleedOut
from pycycle.viewers import print_flow_station, print_compressor, print_turbine, \
                            print_nozzle, print_bleed, print_shaft, print_burner, print_mixer
from pycycle.maps.axi5 import AXI5
from pycycle.maps.lpt2269 import LPT2269
# from pycycle.maps.CFM56_Fan_map import FanMap
# from pycycle.maps.CFM56_HPC_map import HPCmap
from pycycle.maps.CFM56_HPT_map import HPTmap
from pycycle.maps.CFM56_LPT_map import LPTmap


from path_dependent_missions.f110_pycycle.map_msfan3_3 import  FanMap
from path_dependent_missions.f110_pycycle.map_hpc9_3 import  HPCmap

class MixedFlowTurbofan(Group):

    def initialize(self):
        self.options.declare('design', default=True,
            desc='Switch between on-design and off-design calculation.')

    def setup(self):
        thermo_spec = species_data.janaf
        design = self.options['design']

        ##########################################
        # Elements
        ##########################################

        self.add_subsystem('fc', FlightConditions(thermo_data=thermo_spec, elements=AIR_MIX))
        # Inlet Components
        self.add_subsystem('inlet', Inlet(design=design, thermo_data=thermo_spec, elements=AIR_MIX))

        # Fan Components - Split here for CFD integration Add a CFDStart Compomponent
        self.add_subsystem('fan', Compressor(map_data=AXI5, design=design, thermo_data=thermo_spec, elements=AIR_MIX,
                                             map_extrap=True),
                           promotes_inputs=[('Nmech','LP_Nmech')])

        self.add_subsystem('splitter', Splitter(design=design, thermo_data=thermo_spec, elements=AIR_MIX))

        # Bypass Stream
        ###################
        self.add_subsystem('bypass_duct', Duct(design=design, thermo_data=thermo_spec, elements=AIR_MIX))

        # Core Stream
        ##############
        self.add_subsystem('ic_duct', Duct(design=design, thermo_data=thermo_spec, elements=AIR_MIX))

        self.add_subsystem('hpc', Compressor(map_data=HPCmap, design=design, thermo_data=thermo_spec, elements=AIR_MIX,
                                        bleed_names=['cool1', 'cool2'], map_extrap=True),
                            promotes_inputs=[('Nmech','HP_Nmech')])

        self.add_subsystem('duct_3', Duct(design=design, thermo_data=thermo_spec, elements=AIR_MIX))

        self.add_subsystem('bleed_3', BleedOut(design=design, bleed_names=['cust_bleed']))

        self.add_subsystem('burner', Combustor(design=design,thermo_data=thermo_spec,
                                                inflow_elements=AIR_MIX,
                                                air_fuel_elements=AIR_FUEL_MIX,
                                                fuel_type='Jet-A(g)'))
        self.add_subsystem('hpt', Turbine(map_data=HPTmap, design=design, thermo_data=thermo_spec, elements=AIR_FUEL_MIX,
                                          bleed_names=['chargable', 'non_chargable'], map_extrap=True),
                            promotes_inputs=[('Nmech','HP_Nmech')])

        self.add_subsystem('it_duct', Duct(design=design, thermo_data=thermo_spec, elements=AIR_FUEL_MIX))

        # uncooled lpt
        self.add_subsystem('lpt', Turbine(map_data=LPTmap, design=design, thermo_data=thermo_spec, elements=AIR_FUEL_MIX,
                                          map_extrap=True), promotes_inputs=[('Nmech','LP_Nmech')])

        self.add_subsystem('mixer', Mixer(design=design, designed_stream=1,
                                          Fl_I1_elements=AIR_FUEL_MIX, Fl_I2_elements=AIR_MIX))

        # augmentor Components
        self.add_subsystem('augmentor', Combustor(design=design,thermo_data=thermo_spec,
                                                inflow_elements=AIR_FUEL_MIX,
                                                air_fuel_elements=AIR_FUEL_MIX,
                                                fuel_type='Jet-A(g)'))
        # End CFD HERE
        # Nozzle
        self.add_subsystem('nozzle', Nozzle(nozzType='CD', lossCoef='Cfg', thermo_data=thermo_spec, elements=AIR_FUEL_MIX))

        # Mechanical components
        self.add_subsystem('lp_shaft', Shaft(num_ports=2),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('hp_shaft', Shaft(num_ports=2),promotes_inputs=[('Nmech','HP_Nmech')])

        # Aggregating component
        self.add_subsystem('perf', Performance(num_nozzles=1, num_burners=2))


        ##########################################
        #  Connecting the Flow Path
        ##########################################
        connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I')
        connect_flow(self, 'inlet.Fl_O', 'fan.Fl_I')
        connect_flow(self, 'fan.Fl_O', 'splitter.Fl_I')

        # Bypass Connections
        connect_flow(self, 'splitter.Fl_O2', 'bypass_duct.Fl_I')
        connect_flow(self, 'bypass_duct.Fl_O', 'mixer.Fl_I2')

        # Core connections
        connect_flow(self, 'splitter.Fl_O1', 'ic_duct.Fl_I')
        connect_flow(self, 'ic_duct.Fl_O', 'hpc.Fl_I')
        connect_flow(self, 'hpc.Fl_O', 'duct_3.Fl_I')
        connect_flow(self, 'duct_3.Fl_O', 'bleed_3.Fl_I')
        connect_flow(self, 'bleed_3.Fl_O', 'burner.Fl_I')
        connect_flow(self, 'burner.Fl_O', 'hpt.Fl_I')
        connect_flow(self, 'hpt.Fl_O', 'it_duct.Fl_I')
        connect_flow(self, 'it_duct.Fl_O', 'lpt.Fl_I')
        connect_flow(self, 'lpt.Fl_O', 'mixer.Fl_I1')

        connect_flow(self, 'mixer.Fl_O', 'augmentor.Fl_I')
        connect_flow(self,'augmentor.Fl_O','nozzle.Fl_I')

        # Connect cooling flows
        connect_flow(self, 'hpc.cool1', 'hpt.non_chargable', connect_stat=False)
        connect_flow(self, 'hpc.cool2', 'hpt.chargable', connect_stat=False)

        ##########################################
        #  Additional Connections
        ##########################################
        # Make additional model connections
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('augmentor.Wfuel', 'perf.Wfuel_1')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozzle.Fg', 'perf.Fg_0')

        self.connect('fan.trq', 'lp_shaft.trq_0')
        self.connect('lpt.trq', 'lp_shaft.trq_1')

        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')

        self.connect('fc.Fl_O:stat:P', 'nozzle.Ps_exhaust')

        ##########################################
        #  Balances to define cycle convergence
        ##########################################
        balance = self.add_subsystem('balance', BalanceComp())
        if design:
            balance.add_balance('W', lower=1e-3, upper=200., units='lbm/s', eq_units='lbf')
            self.connect('balance.W', 'fc.fs.W')
            self.connect('perf.Fn', 'balance.lhs:W')
            # self.add_subsystem('wDV',IndepVarComp('wDes',100,units='lbm/s'))
            # self.connect('wDV.wDes','fc.fs.W')

            balance.add_balance('BPR', eq_units=None, lower=0.25, val=5.0)
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('mixer.ER', 'balance.lhs:BPR')

            balance.add_balance('FAR_core', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR_core', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR_core')

            balance.add_balance('FAR_ab', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR_ab', 'augmentor.Fl_I:FAR')
            self.connect('augmentor.Fl_O:tot:T', 'balance.lhs:FAR_ab')

            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('lp_shaft.pwr_in', 'balance.lhs:lpt_PR')
            self.connect('lp_shaft.pwr_out', 'balance.rhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_in', 'balance.lhs:hpt_PR')
            self.connect('hp_shaft.pwr_out', 'balance.rhs:hpt_PR')

        else: # off design
            balance.add_balance('W', lower=1e-3, upper=400., units='lbm/s', eq_units='inch**2')
            self.connect('balance.W', 'fc.fs.W')
            self.connect('nozzle.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('BPR', lower=0.25, upper=3.0, eq_units='psi')
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('mixer.Fl_I1_calc:stat:P', 'balance.lhs:BPR')
            self.connect('bypass_duct.Fl_O:stat:P', 'balance.rhs:BPR')

            balance.add_balance('FAR_core', eq_units='degR', lower=1e-4, upper=.045, val=.017)
            self.connect('balance.FAR_core', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR_core')

            balance.add_balance('FAR_ab', eq_units='degR', lower=1e-4, upper=.045, val=.017)
            self.connect('balance.FAR_ab', 'augmentor.Fl_I:FAR')
            self.connect('augmentor.Fl_O:tot:T', 'balance.lhs:FAR_ab')

            balance.add_balance('LP_Nmech', val=1., units='rpm', lower=0.5, upper=2., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.LP_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_in', 'balance.lhs:LP_Nmech')
            self.connect('lp_shaft.pwr_out', 'balance.rhs:LP_Nmech')

            balance.add_balance('HP_Nmech', val=1., units='rpm', lower=0.5, upper=2., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.HP_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_in', 'balance.lhs:HP_Nmech')
            self.connect('hp_shaft.pwr_out', 'balance.rhs:HP_Nmech')

        newton = self.nonlinear_solver = NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-10
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 20
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.linesearch = BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        # newton.linesearch.options['print_bound_enforce'] = True
        newton.linesearch.options['iprint'] = -1

        # newton.linesearch = ArmijoGoldsteinLS()
        # newton.linesearch.options['c'] = -.1

        self.linear_solver = DirectSolver(assemble_jac=True)

        # TODO: re-factor pycycle so this block isn't needed in default use case!!!
        if design:
            ##########################################
            #  Model Parameters
            ##########################################
            element_params = self.add_subsystem('element_params', IndepVarComp(), promotes=["*"])
            element_params.add_output('inlet:ram_recovery', 0.9990)
            element_params.add_output('inlet:MN_out', 0.65)
            self.connect('inlet:ram_recovery', 'inlet.ram_recovery')
            self.connect('inlet:MN_out', 'inlet.MN')

            element_params.add_output('fan:effDes', 0.8700)
            element_params.add_output('fan:MN_out', 0.4578)
            self.connect('fan:effDes', 'fan.map.effDes')
            self.connect('fan:MN_out', 'fan.MN')

            element_params.add_output('splitter:MN_out1', 0.3104)
            element_params.add_output('splitter:MN_out2', 0.4518)
            self.connect('splitter:MN_out1', 'splitter.MN1')
            self.connect('splitter:MN_out2', 'splitter.MN2')

            element_params.add_output('ic_duct:dPqP', 0.0048)
            element_params.add_output('ic_duct:MN_out', 0.3121)
            self.connect('ic_duct:dPqP', 'ic_duct.dPqP')
            self.connect('ic_duct:MN_out', 'ic_duct.MN')

            element_params.add_output('hpc:effDes', 0.8707)
            element_params.add_output('hpc:MN_out', 0.2442)
            self.connect('hpc:effDes', 'hpc.map.effDes')
            self.connect('hpc:MN_out', 'hpc.MN')

            element_params.add_output('hpc:cool1:frac_W', 0.09)
            element_params.add_output('hpc:cool1:frac_P', 1.0)
            element_params.add_output('hpc:cool1:frac_work', 1.0)
            self.connect('hpc:cool1:frac_W', 'hpc.cool1:frac_W')
            self.connect('hpc:cool1:frac_P', 'hpc.cool1:frac_P')
            self.connect('hpc:cool1:frac_work', 'hpc.cool1:frac_work')

            element_params.add_output('hpc:cool2:frac_W', 0.07)
            element_params.add_output('hpc:cool2:frac_P', 0.5)
            element_params.add_output('hpc:cool2:frac_work', 0.5)
            self.connect('hpc:cool2:frac_W', 'hpc.cool2:frac_W')
            self.connect('hpc:cool2:frac_P', 'hpc.cool2:frac_P')
            self.connect('hpc:cool2:frac_work', 'hpc.cool2:frac_work')

            element_params.add_output('duct_3:dPqP', 0.0048)
            element_params.add_output('duct_3:MN_out', 0.2)
            self.connect('duct_3:dPqP', 'duct_3.dPqP')
            self.connect('duct_3:MN_out', 'duct_3.MN')

            element_params.add_output('bleed_3:MN_out', 0.3000)
            element_params.add_output('bleed_3:cust_bleed:frac_W', 0.07)
            self.connect('bleed_3:MN_out', 'bleed_3.MN')
            self.connect('bleed_3:cust_bleed:frac_W', 'bleed_3.cust_bleed:frac_W')

            element_params.add_output('burner:dPqP', 0.0540)
            element_params.add_output('burner:MN_out', 0.1025)
            self.connect('burner:dPqP', 'burner.dPqP')
            self.connect('burner:MN_out', 'burner.MN')

            element_params.add_output('hpt:effDes', 0.8888)
            element_params.add_output('hpt:MN_out', 0.3650)
            element_params.add_output('hpt:chargable:frac_P', 0.0)
            element_params.add_output('hpt:non_chargable:frac_P', 1.0)
            self.connect('hpt:effDes', 'hpt.map.effDes')
            self.connect('hpt:MN_out', 'hpt.MN')
            self.connect('hpt:chargable:frac_P', 'hpt.chargable:frac_P')
            self.connect('hpt:non_chargable:frac_P', 'hpt.non_chargable:frac_P')

            element_params.add_output('it_duct:dPqP', 0.0051)
            element_params.add_output('it_duct:MN_out', 0.3063)
            self.connect('it_duct:dPqP', 'it_duct.dPqP')
            self.connect('it_duct:MN_out', 'it_duct.MN')

            element_params.add_output('lpt:effDes', 0.8996)
            element_params.add_output('lpt:MN_out', 0.4127)
            self.connect('lpt:effDes', 'lpt.map.effDes')
            self.connect('lpt:MN_out', 'lpt.MN')

            element_params.add_output('bypass_duct:dPqP', 0.0107)
            element_params.add_output('bypass_duct:MN_out', 0.4463)
            self.connect('bypass_duct:dPqP', 'bypass_duct.dPqP')
            self.connect('bypass_duct:MN_out', 'bypass_duct.MN')

            # No params for mixer

            element_params.add_output('augmentor:dPqP', 0.0540)
            element_params.add_output('augmentor:MN_out', 0.1025)
            self.connect('augmentor:dPqP', 'augmentor.dPqP')
            self.connect('augmentor:MN_out', 'augmentor.MN')

            element_params.add_output('nozzle:Cfg', 0.9933)
            self.connect('nozzle:Cfg', 'nozzle.Cfg')

            element_params.add_output('lp_shaft:Nmech', 1, units='rpm')
            element_params.add_output('hp_shaft:Nmech', 1, units='rpm')
            element_params.add_output('hp_shaft:HPX', 250.0, units='hp')
            self.connect('lp_shaft:Nmech', 'LP_Nmech')
            self.connect('hp_shaft:Nmech', 'HP_Nmech')
            self.connect('hp_shaft:HPX', 'hp_shaft.HPX')





def print_perf(prob,ptName):
    ''' print out the performancs values'''
    tmpl = 'BPR: {:5.3f}   W: {:5.3f}    Fnet: {:5.3f}    TSFC: {:5.3f}'
    data = [prob[ptName+'.balance.BPR'][0],
            prob[ptName+'.balance.W'][0],
            prob[ptName+'.perf.Fn'][0],
            prob[ptName+'.perf.TSFC'][0]]

    print(tmpl.format(*data))
    print()


def page_viewer(prob,point):
        flow_stations = ['fc.Fl_O', 'inlet.Fl_O', 'fan.Fl_O', 'bypass_duct.Fl_O',
                         'splitter.Fl_O2', 'splitter.Fl_O1', 'ic_duct.Fl_O',
                         'hpc.Fl_O', 'duct_3.Fl_O', 'bleed_3.Fl_O', 'burner.Fl_O', 'hpt.Fl_O',
                         'it_duct.Fl_O', 'lpt.Fl_O', 'mixer.Fl_O', 'augmentor.Fl_O', 'nozzle.Fl_O']

        compressors = ['fan', 'hpc']
        burners = ['burner', 'augmentor']
        turbines = ['hpt', 'lpt']
        shafts = ['hp_shaft', 'lp_shaft']

        MN = prob[point+'.fc.conv.fs.exit_static.statics.ps_resid.MN'][0]
        alt = prob[point+'.fc.ambient.readAtmTable.alt'][0]
        print('*'*100)
        print('* ' + ' '*10 + point + "   MN: {}    alt: {} ft".format(MN, alt))
        print('*'*100)

        print_perf(prob, point)

        print_flow_station(prob,[point+ "."+fl for fl in flow_stations])
        print_compressor(prob,[point+ "." + c for c in compressors])
        print_burner(prob,[point+ "." + b for b in burners])
        print_turbine(prob,[point+ "." + turb for turb in turbines])
        print_mixer(prob, [point+'.'+'mixer'])
        print_nozzle(prob, [point + '.nozzle'])
        print_shaft(prob, [point+ "." + s for s in shafts])
        print_bleed(prob, [point+'.hpc.cool1', point+'.hpc.cool2', point+'.bleed_3.cust_bleed'])


def connect_des_data(prob, des_pt_name, od_pt_names):

    if isinstance(od_pt_names, str):
        od_pt_names = [od_pt_names]

    for pt in od_pt_names:

        prob.model.connect('DESIGN.inlet:ram_recovery', pt+'.inlet.ram_recovery')
        prob.model.connect('DESIGN.nozzle:Cfg', pt+'.nozzle.Cfg')
        prob.model.connect('DESIGN.hp_shaft:HPX', pt+'.hp_shaft.HPX')


        # duct pressure losses
        prob.model.connect('DESIGN.ic_duct:dPqP', pt+'.ic_duct.dPqP')
        prob.model.connect('DESIGN.duct_3:dPqP', pt+'.duct_3.dPqP')
        prob.model.connect('DESIGN.it_duct:dPqP', pt+'.it_duct.dPqP')
        prob.model.connect('DESIGN.bypass_duct:dPqP', pt+'.bypass_duct.dPqP')

        # burner pressure losses
        prob.model.connect('DESIGN.burner:dPqP', pt+'.burner.dPqP')
        prob.model.connect('DESIGN.augmentor:dPqP', pt+'.augmentor.dPqP')

        # cooling flow fractions
        prob.model.connect('DESIGN.hpc:cool1:frac_W', pt+'.hpc.cool1:frac_W')
        prob.model.connect('DESIGN.hpc:cool1:frac_P', pt+'.hpc.cool1:frac_P')
        prob.model.connect('DESIGN.hpc:cool1:frac_work', pt+'.hpc.cool1:frac_work')

        prob.model.connect('DESIGN.hpc:cool2:frac_W', pt+'.hpc.cool2:frac_W')
        prob.model.connect('DESIGN.hpc:cool2:frac_P', pt+'.hpc.cool2:frac_P')
        prob.model.connect('DESIGN.hpc:cool2:frac_work', pt+'.hpc.cool2:frac_work')

        prob.model.connect('DESIGN.bleed_3:cust_bleed:frac_W', pt+'.bleed_3.cust_bleed:frac_W')
        prob.model.connect('DESIGN.hpt:chargable:frac_P', pt+'.hpt.chargable:frac_P')
        prob.model.connect('DESIGN.hpt:non_chargable:frac_P', pt+'.hpt.non_chargable:frac_P')

        # map scalars
        prob.model.connect('DESIGN.fan.s_PRdes', pt+'.fan.s_PRdes')
        prob.model.connect('DESIGN.fan.s_WcDes', pt+'.fan.s_WcDes')
        prob.model.connect('DESIGN.fan.s_effDes', pt+'.fan.s_effDes')
        prob.model.connect('DESIGN.fan.s_NcDes', pt+'.fan.s_NcDes')

        prob.model.connect('DESIGN.hpc.s_PRdes', pt+'.hpc.s_PRdes')
        prob.model.connect('DESIGN.hpc.s_WcDes', pt+'.hpc.s_WcDes')
        prob.model.connect('DESIGN.hpc.s_effDes', pt+'.hpc.s_effDes')
        prob.model.connect('DESIGN.hpc.s_NcDes', pt+'.hpc.s_NcDes')

        prob.model.connect('DESIGN.hpt.s_PRdes', pt+'.hpt.s_PRdes')
        prob.model.connect('DESIGN.hpt.s_WpDes', pt+'.hpt.s_WpDes')
        prob.model.connect('DESIGN.hpt.s_effDes', pt+'.hpt.s_effDes')
        prob.model.connect('DESIGN.hpt.s_NpDes', pt+'.hpt.s_NpDes')

        prob.model.connect('DESIGN.lpt.s_PRdes', pt+'.lpt.s_PRdes')
        prob.model.connect('DESIGN.lpt.s_WpDes', pt+'.lpt.s_WpDes')
        prob.model.connect('DESIGN.lpt.s_effDes', pt+'.lpt.s_effDes')
        prob.model.connect('DESIGN.lpt.s_NpDes', pt+'.lpt.s_NpDes')

        # flow areas
        prob.model.connect('DESIGN.nozzle.Throat:stat:area', pt+'.balance.rhs:W')

        prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
        prob.model.connect('DESIGN.fan.Fl_O:stat:area', pt+'.fan.area')
        prob.model.connect('DESIGN.splitter.Fl_O1:stat:area', pt+'.splitter.area1')
        prob.model.connect('DESIGN.splitter.Fl_O2:stat:area', pt+'.splitter.area2')
        prob.model.connect('DESIGN.ic_duct.Fl_O:stat:area', pt+'.ic_duct.area')
        prob.model.connect('DESIGN.hpc.Fl_O:stat:area', pt+'.hpc.area')
        prob.model.connect('DESIGN.duct_3.Fl_O:stat:area', pt+'.duct_3.area')
        prob.model.connect('DESIGN.bleed_3.Fl_O:stat:area', pt+'.bleed_3.area')
        prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
        prob.model.connect('DESIGN.hpt.Fl_O:stat:area', pt+'.hpt.area')
        prob.model.connect('DESIGN.it_duct.Fl_O:stat:area', pt+'.it_duct.area')
        prob.model.connect('DESIGN.lpt.Fl_O:stat:area', pt+'.lpt.area')
        prob.model.connect('DESIGN.bypass_duct.Fl_O:stat:area', pt+'.bypass_duct.area')

        prob.model.connect('DESIGN.mixer.Fl_O:stat:area', pt+'.mixer.area')
        prob.model.connect('DESIGN.mixer.Fl_I1_calc:stat:area', pt+'.mixer.Fl_I1_stat_calc.area')

        prob.model.connect('DESIGN.augmentor.Fl_O:stat:area', pt+'.augmentor.area')


if __name__ == "__main__":

    import time
    from openmdao.api import Problem

    prob = Problem()

    des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=["*"])

    ##########################################
    #  Design Variables
    ##########################################
    des_vars.add_output('alt', 0., units='ft') #DV
    des_vars.add_output('MN', 0.001) #DV
    des_vars.add_output('T4max', 3200, units='degR')
    des_vars.add_output('T4maxab', 3400, units='degR')
    des_vars.add_output('Fn_des', 17000., units='lbf')
    des_vars.add_output('Mix_ER', 1.05 ,units=None) # defined as 1 over 2
    des_vars.add_output('fan:PRdes', 3.3) #ADV
    des_vars.add_output('hpc:PRdes', 9.3)

    #####################
    # DESIGN CASE
    #####################

    prob.model.add_subsystem('DESIGN', MixedFlowTurbofan(design=True))

    prob.model.connect('alt', 'DESIGN.fc.alt')
    prob.model.connect('MN', 'DESIGN.fc.MN')
    prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
    prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR_core')
    prob.model.connect('T4maxab', 'DESIGN.balance.rhs:FAR_ab')
    prob.model.connect('Mix_ER', 'DESIGN.balance.rhs:BPR')

    prob.model.connect('fan:PRdes', 'DESIGN.fan.map.PRdes')

    prob.model.connect('hpc:PRdes', 'DESIGN.hpc.map.PRdes')


    ####################
    # OFF DESIGN CASES
    ####################
    od_pts = ['OD0',]
    # od_pts = []

    od_alts = [0,]
    od_MNs = [0.001, ]

    des_vars.add_output('OD:alts', val=od_alts, units='ft')
    des_vars.add_output('OD:MNs', val=od_MNs)

    connect_des_data(prob, 'DESIGN', od_pts)

    for i,pt in enumerate(od_pts):
        prob.model.add_subsystem(pt, MixedFlowTurbofan(design=False))

        prob.model.connect('OD:alts', pt+'.fc.alt', src_indices=[i,])
        prob.model.connect('OD:MNs', pt+'.fc.MN', src_indices=[i,])

        prob.model.connect('T4max', pt+'.balance.rhs:FAR_core')
        prob.model.connect('T4maxab', pt+'.balance.rhs:FAR_ab')


    # setup problem
    prob.setup(check=False)#True)

    # initial guesses
    prob['DESIGN.balance.FAR_core'] = 0.025
    prob['DESIGN.balance.FAR_ab'] = 0.025
    prob['DESIGN.balance.BPR'] = 0.85
    prob['DESIGN.balance.W'] = 150.
    prob['DESIGN.balance.lpt_PR'] = 3.5
    prob['DESIGN.balance.hpt_PR'] = 2.5
    prob['DESIGN.fc.balance.Pt'] = 14.
    prob['DESIGN.fc.balance.Tt'] = 500.0
    prob['DESIGN.mixer.balance.P_tot']= 500.

    for pt in od_pts:
        prob[pt+'.balance.FAR_core'] = 0.028
        prob[pt+'.balance.FAR_ab'] = 0.034
        prob[pt+'.balance.BPR'] = .9
        prob[pt+'.balance.W'] = 157.225
        prob[pt+'.balance.HP_Nmech'] = 1.
        prob[pt+'.balance.LP_Nmech'] = 1.
        prob[pt+'.fc.balance.Pt'] = 14.696
        prob[pt+'.fc.balance.Tt'] = 518.67
        prob[pt+'.mixer.balance.P_tot'] = 380.
        prob[pt+'.hpt.PR'] = 3.439
        prob[pt+'.lpt.PR'] = 2.438
        prob[pt+'.fan.map.RlineMap'] = 2.0
        prob[pt+'.hpc.map.RlineMap'] = 2.0

    st = time.time()

    # prob.model.DESIGN.nonlinear_solver.options['maxiter'] = 3
    # prob.model.OD0.nonlinear_solver.options['maxiter'] = 3
    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    prob.run_model()

    # prob.model.DESIGN.list_outputs(residuals=True, units=True, residuals_tol=1e-3)
    # prob.model.OD0.list_outputs(residuals=True, units=True, residuals_tol=1e-4)
    # prob.model.OD0.list_outputs(residuals=True, units=True)
    # prob.model.OD0.mixer.list_outputs(residuals=True, units=True, residuals_tol=1e-3)
    # prob.model.DESIGN.mixer.Fl_I1_stat_calc.list_inputs()
    # prob.model.DESIGN.mixer.Fl_I1_stat_calc.list_outputs()
    # print()
    # prob.model.OD0.mixer.Fl_I1_stat_calc.list_inputs(print_arrays=True)
    # prob.model.OD0.mixer.Fl_I1_stat_calc.list_outputs()
    # exit()



    page_viewer(prob, 'DESIGN')
    print(3*"\n")
    page_viewer(prob,'OD0')
    print()
    print("time", time.time() - st)

