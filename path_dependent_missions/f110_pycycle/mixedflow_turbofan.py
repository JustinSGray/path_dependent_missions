from __future__ import print_function

import sys
import numpy as np

from openmdao.api import ImplicitComponent, Group, IndepVarComp, BalanceComp, DirectSolver, BoundsEnforceLS, NewtonSolver
from openmdao.core.component import Component

from pycycle.constants import AIR_MIX, AIR_FUEL_MIX
from pycycle.connect_flow import connect_flow
# from pycycle.balance import Balance
from pycycle.cea import species_data
from pycycle.elements.api import FlightConditions, Inlet, Compressor, Combustor, Turbine, Nozzle, Shaft, Duct, Performance,Splitter, Mixer, BleedOut
from pycycle.viewers import print_flow_station, print_compressor, print_turbine, \
                            print_nozzle, print_bleed, print_shaft, print_burner
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
        self.add_subsystem('inlet_duct', Duct(design=design, thermo_data=thermo_spec, elements=AIR_MIX))
        # Fan Components - Split here for CFD integration Add a CFDStart Compomponent
        self.add_subsystem('fan', Compressor(map_data=AXI5, design=design, thermo_data=thermo_spec, elements=AIR_MIX,
                                             map_extrap=True),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('splitter', Splitter(design=design, thermo_data=thermo_spec, elements=AIR_MIX))
        # Core Stream components
        self.add_subsystem('splitter_core_duct', Duct(design=design, thermo_data=thermo_spec, elements=AIR_MIX))

        self.add_subsystem('hpc', Compressor(map_data=HPCmap, design=design, thermo_data=thermo_spec, elements=AIR_MIX,
                                        bleed_names=['cool1'],map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld3', BleedOut(design=design, bleed_names=['cool3']))
        self.add_subsystem('burner', Combustor(design=design,thermo_data=thermo_spec,
                                                inflow_elements=AIR_MIX,
                                                air_fuel_elements=AIR_FUEL_MIX,
                                                fuel_type='Jet-A(g)'))
        self.add_subsystem('hpt', Turbine(map_data=HPTmap, design=design, thermo_data=thermo_spec, elements=AIR_FUEL_MIX,
                                          bleed_names=['cool3'],map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('hpt_duct', Duct(design=design, thermo_data=thermo_spec, elements=AIR_FUEL_MIX))
        self.add_subsystem('lpt', Turbine(map_data=LPTmap, design=design, thermo_data=thermo_spec, elements=AIR_FUEL_MIX,
                                        bleed_names=['cool1'],map_extrap=True), promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('lpt_duct', Duct(design=design, thermo_data=thermo_spec, elements=AIR_FUEL_MIX))
        # Bypass Components
        self.add_subsystem('bypass_duct', Duct(design=design, thermo_data=thermo_spec, elements=AIR_MIX))
        # Mixer component
        self.add_subsystem('mixer', Mixer(design=design, designed_stream=1,
                                          Fl_I1_elements=AIR_FUEL_MIX, Fl_I2_elements=AIR_MIX))
        self.add_subsystem('mixer_duct', Duct(design=design, thermo_data=thermo_spec, elements=AIR_FUEL_MIX))
        # Afterburner Components
        self.add_subsystem('afterburner', Combustor(design=design,thermo_data=thermo_spec,
                                                inflow_elements=AIR_FUEL_MIX,
                                                air_fuel_elements=AIR_FUEL_MIX,
                                                fuel_type='Jet-A(g)'))
        # End CFD HERE
        # Nozzle
        self.add_subsystem('mixed_nozz', Nozzle(nozzType='CD', lossCoef='Cfg', thermo_data=thermo_spec, elements=AIR_FUEL_MIX))

        # Mechanical components
        self.add_subsystem('lp_shaft', Shaft(num_ports=2),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('hp_shaft', Shaft(num_ports=2),promotes_inputs=[('Nmech','HP_Nmech')])

        # Aggregating component
        self.add_subsystem('perf', Performance(num_nozzles=1, num_burners=2))


        ##########################################
        #  Connecting the Flow Path
        ##########################################
        connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I')
        connect_flow(self, 'inlet.Fl_O', 'inlet_duct.Fl_I')
        connect_flow(self, 'inlet_duct.Fl_O', 'fan.Fl_I')
        connect_flow(self, 'fan.Fl_O', 'splitter.Fl_I')
        # Core connections
        connect_flow(self, 'splitter.Fl_O1', 'splitter_core_duct.Fl_I')
        connect_flow(self, 'splitter_core_duct.Fl_O', 'hpc.Fl_I')
        connect_flow(self, 'hpc.Fl_O', 'bld3.Fl_I')
        connect_flow(self, 'bld3.Fl_O', 'burner.Fl_I')
        connect_flow(self, 'burner.Fl_O', 'hpt.Fl_I')
        connect_flow(self, 'hpt.Fl_O', 'hpt_duct.Fl_I')
        connect_flow(self, 'hpt_duct.Fl_O', 'lpt.Fl_I')
        connect_flow(self, 'lpt.Fl_O', 'lpt_duct.Fl_I')
        connect_flow(self, 'lpt_duct.Fl_O','mixer.Fl_I1')
        # Bypass Connections
        connect_flow(self, 'splitter.Fl_O2', 'bypass_duct.Fl_I')
        connect_flow(self, 'bypass_duct.Fl_O', 'mixer.Fl_I2')
        #Mixer Connections
        connect_flow(self, 'mixer.Fl_O', 'mixer_duct.Fl_I')
        # After Burner
        connect_flow(self,'mixer_duct.Fl_O','afterburner.Fl_I')
        # Nozzle
        connect_flow(self,'afterburner.Fl_O','mixed_nozz.Fl_I')
        # Connect cooling flows
        connect_flow(self, 'hpc.cool1', 'lpt.cool1', connect_stat=False)
        connect_flow(self, 'bld3.cool3', 'hpt.cool3', connect_stat=False)

        ##########################################
        #  Additional Connections
        ##########################################
        # Make additional model connections
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('afterburner.Wfuel', 'perf.Wfuel_1')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('mixed_nozz.Fg', 'perf.Fg_0')

        self.connect('fan.trq', 'lp_shaft.trq_0')
        self.connect('lpt.trq', 'lp_shaft.trq_1')
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')
        self.connect('fc.Fl_O:stat:P', 'mixed_nozz.Ps_exhaust')

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
            self.connect('balance.FAR_ab', 'afterburner.Fl_I:FAR')
            self.connect('afterburner.Fl_O:tot:T', 'balance.lhs:FAR_ab')

            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0., res_ref=1)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0., res_ref=1)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:hpt_PR')

        else: # off design
            balance.add_balance('W', lower=1e-3, upper=200., units='lbm/s', eq_units='inch**2')
            self.connect('balance.W', 'fc.fs.W')
            self.connect('mixed_nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('BPR', lower=0.25, upper=5.0, eq_units='psi')
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('mixer.Fl_I1_calc:stat:P', 'balance.lhs:BPR')
            self.connect('bypass_duct.Fl_O:stat:P', 'balance.rhs:BPR')

            balance.add_balance('FAR_core', eq_units='degR', lower=1e-4, upper=.06, val=.017)
            self.connect('balance.FAR_core', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR_core')

            balance.add_balance('FAR_ab', eq_units='degR', lower=1e-4, upper=.06, val=.017)
            self.connect('balance.FAR_ab', 'afterburner.Fl_I:FAR')
            self.connect('afterburner.Fl_O:tot:T', 'balance.lhs:FAR_ab')

            balance.add_balance('lp_Nmech', val=1000., units='rpm', lower=500., eq_units='hp', rhs_val=0., res_ref=1e3)
            self.connect('balance.lp_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:lp_Nmech')

            balance.add_balance('hp_Nmech', val=1000., units='rpm', lower=500., eq_units='hp', rhs_val=0., res_ref=1e3)
            self.connect('balance.hp_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:hp_Nmech')

        newton = self.nonlinear_solver = NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-10
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.linesearch = BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1


        self.linear_solver = DirectSolver(assemble_jac=True)

def print_perf(prob,ptName):
    ''' print out the performancs values'''
    print('BPR',prob[ptName+'.balance.BPR'])
    print('W',prob[ptName+'.balance.W'])
    #print('W',prob[ptName+'.wDV.wDes'])
    print('Fnet uninst.',prob[ptName+'.perf.Fn'])

if __name__ == "__main__":
    import time
    from openmdao.api import Problem

    prob = Problem()

    des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=["*"])

    ##########################################
    #  Design Variables
    ##########################################
    des_vars.add_output('alt', 35000., units='ft') #DV
    des_vars.add_output('MN', 0.8) #DV
    des_vars.add_output('T4max', 3200, units='degR')
    des_vars.add_output('T4maxab', 3400, units='degR')
    des_vars.add_output('Fn_des', 5500.0, units='lbf')
    des_vars.add_output('Mix_ER', 1.05 ,units=None) # defined as 1 over 2
    des_vars.add_output('fan:PRdes', 3.3) #ADV
    des_vars.add_output('hpc:PRdes', 9.3)


    ##########################################
    #  Model Parameters
    ##########################################
    element_params = prob.model.add_subsystem('element_params', IndepVarComp(), promotes=["*"])
    element_params.add_output('inlet:ram_recovery', 0.9990)
    element_params.add_output('inlet:MN_out', 0.751)

    element_params.add_output('inlet_duct:dPqP', 0.0)
    element_params.add_output('inlet_duct:MN_out', 0.4)

    element_params.add_output('fan:effDes', 0.8700)
    element_params.add_output('fan:MN_out', 0.4578)

    element_params.add_output('splitter:MN_out1', 0.3104)
    element_params.add_output('splitter:MN_out2', 0.4518)

    element_params.add_output('splitter_core_duct:dPqP', 0.0048)
    element_params.add_output('splitter_core_duct:MN_out', 0.3121)

    element_params.add_output('hpc:effDes', 0.8707)
    element_params.add_output('hpc:MN_out', 0.2442)

    element_params.add_output('bld3:MN_out', 0.3000)

    element_params.add_output('burner:dPqP', 0.0540)
    element_params.add_output('burner:MN_out', 0.1025)

    element_params.add_output('hpt:effDes', 0.8888)
    element_params.add_output('hpt:MN_out', 0.3650)

    element_params.add_output('hpt_duct:dPqP', 0.0051)
    element_params.add_output('hpt_duct:MN_out', 0.3063)

    element_params.add_output('lpt:effDes', 0.8996)
    element_params.add_output('lpt:MN_out', 0.4127)

    element_params.add_output('lpt_duct:dPqP', 0.0107)
    element_params.add_output('lpt_duct:MN_out', 0.4463)

    element_params.add_output('bypass_duct:dPqP', 0.0107)
    element_params.add_output('bypass_duct:MN_out', 0.4463)

    # No params for mixer

    element_params.add_output('mixer_duct:dPqP', 0.0107)
    element_params.add_output('mixer_duct:MN_out', 0.4463)

    element_params.add_output('afterburner:dPqP', 0.0540)
    element_params.add_output('afterburner:MN_out', 0.1025)

    element_params.add_output('mixed_nozz:Cfg', 0.9933)

    element_params.add_output('lp_shaft:Nmech', 4666.1, units='rpm')
    element_params.add_output('hp_shaft:Nmech', 14705.7, units='rpm')
    element_params.add_output('hp_shaft:HPX', 250.0, units='hp')

    element_params.add_output('hpc:cool1:frac_W', 0.050708)
    element_params.add_output('hpc:cool1:frac_P', 0.5)
    element_params.add_output('hpc:cool1:frac_work', 0.5)

    element_params.add_output('bld3:cool3:frac_W', 0.067214)

    element_params.add_output('hpt:cool3:frac_P', 1.0)
    element_params.add_output('lpt:cool1:frac_P', 1.0)

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

    prob.model.connect('inlet:ram_recovery', 'DESIGN.inlet.ram_recovery')
    prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')

    prob.model.connect('inlet_duct:dPqP', 'DESIGN.inlet_duct.dPqP')
    prob.model.connect('inlet_duct:MN_out', 'DESIGN.inlet_duct.MN')

    prob.model.connect('fan:PRdes', 'DESIGN.fan.map.PRdes')
    prob.model.connect('fan:effDes', 'DESIGN.fan.map.effDes')
    prob.model.connect('fan:MN_out', 'DESIGN.fan.MN')

    #prob.model.connect('splitter:BPR', 'DESIGN.splitter.BPR')
    prob.model.connect('splitter:MN_out1', 'DESIGN.splitter.MN1')
    prob.model.connect('splitter:MN_out2', 'DESIGN.splitter.MN2')

    prob.model.connect('splitter_core_duct:dPqP', 'DESIGN.splitter_core_duct.dPqP')
    prob.model.connect('splitter_core_duct:MN_out', 'DESIGN.splitter_core_duct.MN')

    prob.model.connect('hpc:PRdes', 'DESIGN.hpc.map.PRdes')
    prob.model.connect('hpc:effDes', 'DESIGN.hpc.map.effDes')
    prob.model.connect('hpc:MN_out', 'DESIGN.hpc.MN')

    prob.model.connect('bld3:MN_out', 'DESIGN.bld3.MN')

    prob.model.connect('burner:dPqP', 'DESIGN.burner.dPqP')
    prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')

    prob.model.connect('hpt:effDes', 'DESIGN.hpt.map.effDes')
    prob.model.connect('hpt:MN_out', 'DESIGN.hpt.MN')

    prob.model.connect('hpt_duct:dPqP', 'DESIGN.hpt_duct.dPqP')
    prob.model.connect('hpt_duct:MN_out', 'DESIGN.hpt_duct.MN')

    prob.model.connect('lpt:effDes', 'DESIGN.lpt.map.effDes')
    prob.model.connect('lpt:MN_out', 'DESIGN.lpt.MN')

    prob.model.connect('lpt_duct:dPqP', 'DESIGN.lpt_duct.dPqP')
    prob.model.connect('lpt_duct:MN_out', 'DESIGN.lpt_duct.MN')

    prob.model.connect('bypass_duct:dPqP', 'DESIGN.bypass_duct.dPqP')
    prob.model.connect('bypass_duct:MN_out', 'DESIGN.bypass_duct.MN')

    prob.model.connect('mixer_duct:dPqP', 'DESIGN.mixer_duct.dPqP')
    prob.model.connect('mixer_duct:MN_out', 'DESIGN.mixer_duct.MN')

    prob.model.connect('afterburner:dPqP', 'DESIGN.afterburner.dPqP')
    prob.model.connect('afterburner:MN_out', 'DESIGN.afterburner.MN')

    prob.model.connect('mixed_nozz:Cfg', 'DESIGN.mixed_nozz.Cfg')

    prob.model.connect('lp_shaft:Nmech', 'DESIGN.LP_Nmech')
    prob.model.connect('hp_shaft:Nmech', 'DESIGN.HP_Nmech')
    prob.model.connect('hp_shaft:HPX', 'DESIGN.hp_shaft.HPX')

    prob.model.connect('hpc:cool1:frac_W', 'DESIGN.hpc.cool1:frac_W')
    prob.model.connect('hpc:cool1:frac_P', 'DESIGN.hpc.cool1:frac_P')
    prob.model.connect('hpc:cool1:frac_work', 'DESIGN.hpc.cool1:frac_work')

    prob.model.connect('bld3:cool3:frac_W', 'DESIGN.bld3.cool3:frac_W')

    prob.model.connect('hpt:cool3:frac_P', 'DESIGN.hpt.cool3:frac_P')
    prob.model.connect('lpt:cool1:frac_P', 'DESIGN.lpt.cool1:frac_P')


    ####################
    # OFF DESIGN CASES
    ####################
    od_pts = ['OD0',]
    # od_pts = []

    od_alts = [35000,]
    od_MNs = [0.8, ]

    des_vars.add_output('OD:alts', val=od_alts, units='ft')
    des_vars.add_output('OD:MNs', val=od_MNs)


    for i,pt in enumerate(od_pts):
        prob.model.add_subsystem(pt, MixedFlowTurbofan(design=False))

        prob.model.connect('OD:alts', pt+'.fc.alt', src_indices=[i,])
        prob.model.connect('OD:MNs', pt+'.fc.MN', src_indices=[i,])

        prob.model.connect('T4max', pt+'.balance.rhs:FAR_core')
        prob.model.connect('T4maxab', pt+'.balance.rhs:FAR_ab')

        prob.model.connect('inlet:ram_recovery', pt+'.inlet.ram_recovery')
        prob.model.connect('mixed_nozz:Cfg', pt+'.mixed_nozz.Cfg')
        prob.model.connect('hp_shaft:HPX', pt+'.hp_shaft.HPX')


        # duct pressure losses
        prob.model.connect('inlet_duct:dPqP', pt+'.inlet_duct.dPqP')
        prob.model.connect('splitter_core_duct:dPqP', pt+'.splitter_core_duct.dPqP')
        prob.model.connect('bypass_duct:dPqP', pt+'.bypass_duct.dPqP')
        prob.model.connect('hpt_duct:dPqP', pt+'.hpt_duct.dPqP')
        prob.model.connect('lpt_duct:dPqP', pt+'.lpt_duct.dPqP')
        prob.model.connect('mixer_duct:dPqP', pt+'.mixer_duct.dPqP')

        # burner pressure losses
        prob.model.connect('burner:dPqP', pt+'.burner.dPqP')
        prob.model.connect('afterburner:dPqP', pt+'.afterburner.dPqP')

        # cooling flow fractions
        prob.model.connect('hpc:cool1:frac_W', pt+'.hpc.cool1:frac_W')
        prob.model.connect('hpc:cool1:frac_P', pt+'.hpc.cool1:frac_P')
        prob.model.connect('hpc:cool1:frac_work', pt+'.hpc.cool1:frac_work')
        prob.model.connect('bld3:cool3:frac_W', pt+'.bld3.cool3:frac_W')
        prob.model.connect('hpt:cool3:frac_P', pt+'.hpt.cool3:frac_P')
        prob.model.connect('lpt:cool1:frac_P', pt+'.lpt.cool1:frac_P')

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
        prob.model.connect('DESIGN.mixed_nozz.Throat:stat:area', pt+'.balance.rhs:W')

        prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
        prob.model.connect('DESIGN.fan.Fl_O:stat:area', pt+'.fan.area')
        prob.model.connect('DESIGN.splitter.Fl_O1:stat:area', pt+'.splitter.area1')
        prob.model.connect('DESIGN.splitter.Fl_O2:stat:area', pt+'.splitter.area2')
        prob.model.connect('DESIGN.splitter_core_duct.Fl_O:stat:area', pt+'.splitter_core_duct.area')
        prob.model.connect('DESIGN.hpc.Fl_O:stat:area', pt+'.hpc.area')
        prob.model.connect('DESIGN.bld3.Fl_O:stat:area', pt+'.bld3.area')
        prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
        prob.model.connect('DESIGN.hpt.Fl_O:stat:area', pt+'.hpt.area')
        prob.model.connect('DESIGN.hpt_duct.Fl_O:stat:area', pt+'.hpt_duct.area')
        prob.model.connect('DESIGN.lpt.Fl_O:stat:area', pt+'.lpt.area')
        prob.model.connect('DESIGN.lpt_duct.Fl_O:stat:area', pt+'.lpt_duct.area')
        prob.model.connect('DESIGN.bypass_duct.Fl_O:stat:area', pt+'.bypass_duct.area')
        prob.model.connect('DESIGN.mixer.Fl_O:stat:area', pt+'.mixer.area')
        prob.model.connect('DESIGN.mixer.Fl_I1_calc:stat:area', pt+'.mixer.Fl_I1_stat_calc.area')
        prob.model.connect('DESIGN.mixer_duct.Fl_O:stat:area', pt+'.mixer_duct.area')
        prob.model.connect('DESIGN.afterburner.Fl_O:stat:area', pt+'.afterburner.area')


    # setup problem
    prob.setup(check=False)#True)

    # initial guesses
    prob['DESIGN.balance.FAR_core'] = 0.025
    prob['DESIGN.balance.FAR_ab'] = 0.025
    prob['DESIGN.balance.BPR'] = 1.0
    prob['DESIGN.balance.W'] = 100.
    prob['DESIGN.balance.lpt_PR'] = 3.5
    prob['DESIGN.balance.hpt_PR'] = 2.5
    prob['DESIGN.fc.balance.Pt'] = 5.2
    prob['DESIGN.fc.balance.Tt'] = 440.0
    prob['DESIGN.mixer.balance.P_tot']=100

    for pt in od_pts:
        prob[pt+'.balance.FAR_core'] = 0.031
        prob[pt+'.balance.FAR_ab'] = 0.038
        prob[pt+'.balance.BPR'] = 2.2
        prob[pt+'.balance.W'] = 60
        prob[pt+'.balance.hp_Nmech'] = 14700
        prob[pt+'.balance.lp_Nmech'] = 4866
        prob[pt+'.fc.balance.Pt'] = 5.2
        prob[pt+'.fc.balance.Tt'] = 440.0
        prob[pt+'.mixer.balance.P_tot']=150
        prob[pt+'.hpt.PR'] = 2.5
        prob[pt+'.lpt.PR'] = 3.5
        prob[pt+'.fan.map.RlineMap'] = 2.0
        prob[pt+'.hpc.map.RlineMap'] = 2.0

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    prob.run_model()

    prob.model.DESIGN.list_outputs(residuals=True, units=True, residuals_tol=1e-3)
    exit()

    def page_viewer(point):
        flow_stations = ['fc.Fl_O', 'inlet.Fl_O', 'inlet_duct.Fl_O', 'fan.Fl_O', 'bypass_duct.Fl_O',
                         'splitter.Fl_O2', 'splitter.Fl_O1', 'splitter_core_duct.Fl_O',
                         'hpc.Fl_O', 'bld3.Fl_O', 'burner.Fl_O',
                         'hpt.Fl_O', 'hpt_duct.Fl_O', 'lpt_duct.Fl_O',
                         'mixer.Fl_O', 'mixer_duct.Fl_O', 'afterburner.Fl_O', 'mixed_nozz.Fl_O']

        compressors = ['fan', 'hpc']
        burners = ['burner', 'afterburner']
        turbines = ['hpt', 'lpt']
        shafts = ['hp_shaft', 'lp_shaft']

        print('*'*80)
        print('* ' + ' '*10 + point)
        print('*'*80)

        print_flow_station(prob,[point+ "."+fl for fl in flow_stations])
        print_compressor(prob,[point+ "." + c for c in compressors])
        print_burner(prob,[point+ "." + b for b in burners])
        print_turbine(prob,[point+ "." + turb for turb in turbines])
        print_nozzle(prob, [point + '.mixed_nozz'])
        print_shaft(prob, [point+ "." + s for s in shafts])
        print_bleed(prob, [point+'.hpc.cool1', point+'.bld3.cool3'])

    page_viewer('DESIGN')
    page_viewer('OD0')
    print()
    print("time", time.time() - st)

