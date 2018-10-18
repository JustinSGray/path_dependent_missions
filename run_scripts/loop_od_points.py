from pprint import pprint as pp

from openmdao.api import Problem, IndepVarComp

from path_dependent_missions.f110_pycycle import mixedflow_turbofan as mftf


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

prob.model.add_subsystem('DESIGN', mftf.MixedFlowTurbofan(design=True))

prob.model.connect('alt', 'DESIGN.fc.alt')
prob.model.connect('MN', 'DESIGN.fc.MN')
prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR_core')
prob.model.connect('T4maxab', 'DESIGN.balance.rhs:FAR_ab')
prob.model.connect('Mix_ER', 'DESIGN.balance.rhs:BPR')

prob.model.connect('fan:PRdes', 'DESIGN.fan.map.PRdes')

prob.model.connect('hpc:PRdes', 'DESIGN.hpc.map.PRdes')


######################
# ONE OFF DESIGN CASE
######################


des_vars.add_output('OD:alt', val=0.0, units='ft')
des_vars.add_output('OD:MN', val=0.001)

mftf.connect_des_data(prob, 'DESIGN', 'OD')

prob.model.add_subsystem('OD', mftf.MixedFlowTurbofan(design=False))

prob.model.connect('OD:alt', 'OD.fc.alt')
prob.model.connect('OD:MN', 'OD.fc.MN')

prob.model.connect('T4max', 'OD.far_core_bal.T_requested')
prob.model.connect('T4maxab', 'OD.balance.rhs:FAR_ab')


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
prob['DESIGN.mixer.balance.P_tot']= 72.

prob['OD.far_core_bal.FAR'] = 0.028
prob['OD.balance.FAR_ab'] = 0.034
prob['OD.balance.BPR'] = .9
prob['OD.balance.W'] = 157.225
prob['OD.balance.HP_Nmech'] = 1.
prob['OD.balance.LP_Nmech'] = 1.
prob['OD.fc.balance.Pt'] = 14.696
prob['OD.fc.balance.Tt'] = 518.67
prob['OD.mixer.balance.P_tot'] = 72. # 380.
prob['OD.hpt.PR'] = 3.439
prob['OD.lpt.PR'] = 2.438
prob['OD.fan.map.RlineMap'] = 2.0
prob['OD.hpc.map.RlineMap'] = 2.0


prob.set_solver_print(level=-1)
prob.set_solver_print(level=2, depth=1)

# OD CASES (MN, alt) #each row is one altitude, but data does not necessarily need to be structured
# OD_CASES = [
#     [(0.001, 0.0),(.2, 0.0),(.4, 0.0),(.6, 0.0),(.8, 0.0),(1.0, 0.0),(1.2, 0.0),(1.4, 0.0),(1.6, 0.0),],
#     [(0.001, 1000.),(.2, 1000.),(.4, 1000.),(.6, 1000.),(.8, 1000.),(1.0, 1000.),(1.2, 1000.),(1.4, 1000.),(1.6, 1000.),],
#     [(0.001, 5000.),(.2, 5000.),(.4, 5000.),(.6, 5000.),(.8, 5000.),(1.0, 5000.),(1.2, 5000.),(1.4, 5000.),(1.6, 5000.),],
#     [(0.001, 10000.),(.2, 10000.),(.4, 10000.),(.6, 10000.),(.8, 10000.),(1.0, 10000.),(1.2, 10000.),(1.4, 10000.),(1.6, 10000.),],
# ]

OD_CASES = [
    [(0.001, 0.0),(.2, 0.0)],
    [(0.001, 1000.),(.2, 1000.)],
    [(0.001, 5000.),(.2, 5000.)],
    [(0.001, 10000.),(.2, 10000.)],
    [(0.001, 15000.),(.2, 15000.)],
    [(0.001, 17000.),(.2, 17000.)],
    [(0.2, 20000.),(.4, 20000.)],
    [(0.2, 25000.),(.4, 25000.)],
]



#save data for plotting purposes
data = {
    'MN':[],
    'alt':[],
    'FAR_ab': [],
    'FAR_core': [],
    'Fnet': [],
    'Fram': [],
    'W':[],
    'BPR':[],
    'hp_nmech':[],
    'lp_nmech':[],
    'NPR':[],
    'hpc_eff':[],
    'fan_eff':[],
    'hpc_Nc':[],
    'fan_Nc':[],
}


guess_vars = ['OD.far_core_bal.FAR', 'OD.balance.FAR_ab', 'OD.balance.BPR', 'OD.balance.W', 'OD.balance.HP_Nmech', 'OD.balance.LP_Nmech',
              'OD.fc.balance.Pt', 'OD.fc.balance.Tt', 'OD.mixer.balance.P_tot', 'OD.hpt.PR', 'OD.lpt.PR', 'OD.fan.map.RlineMap', 'OD.hpc.map.RlineMap']

for i, row in enumerate(OD_CASES):
    for j, (MN, alt) in enumerate(row):
        prob['OD:MN'] = MN
        prob['OD:alt'] = alt

        print('\n\n\n\n### MN: {} alt {}'.format(MN, alt))
        prob.run_model()
        if i == 0 and j==0:
            mftf.page_viewer(prob, 'DESIGN')
        mftf.page_viewer(prob, 'OD')

        # save the data
        # FAR are nasty to compute... need to make this easier
        W_fuel = prob['OD.augmentor.Wfuel'][0]
        W_tot = prob['OD.augmentor.Fl_O:stat:W'][0]
        W_air = W_tot - W_fuel
        FAR = W_fuel/W_air
        data['FAR_ab'].append(FAR)

        W_fuel = prob['OD.burner.Wfuel'][0]
        W_tot = prob['OD.burner.Fl_O:stat:W'][0]
        W_air = W_tot - W_fuel
        FAR = W_fuel/W_air
        data['FAR_core'].append(FAR)

        data['Fnet'].append(prob['OD.perf.Fn'][0])
        data['Fram'].append(prob['OD.inlet.F_ram'][0])
        data['W'].append(prob['OD.balance.W'][0])
        data['BPR'].append(prob['OD.balance.BPR'][0])
        data['hp_nmech'].append(prob['OD.hp_shaft.Nmech'][0])
        data['lp_nmech'].append(prob['OD.lp_shaft.Nmech'][0])
        data['NPR'].append(prob['OD.nozzle.PR'][0])
        data['hpc_eff'].append(prob['OD.hpc.eff'][0])
        data['fan_eff'].append(prob['OD.fan.eff'][0])
        data['hpc_Nc'].append(prob['OD.hpc.map.readMap.NcMap'])
        data['fan_Nc'].append(prob['OD.fan.map.readMap.NcMap'])


        data['MN'].append(MN)
        data['alt'].append(alt)

        if i == 0:
            #save_guesses

            guesses = {}
            for var in guess_vars:
                guesses[var] = prob[var]

    if i > 0: # after the first set of altitude cases, reset the guesses back to MN = 0
        for var in guess_vars:
            prob[var] = guesses[var]
        prob['OD.balance.W'] = guesses['OD.balance.W']*.75 #ditry hack, cause altitude is always increasing so mass flow should always decrease

pp(data)



