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


####################
# OFF DESIGN CASES
####################
od_pts = ['OD_DES_CHECK','OD0', 'OD1']
# od_pts = []

od_alts = [0,     0,   20000]
od_MNs =  [0.001, 0.5, 0.5]

des_vars.add_output('OD:alts', val=od_alts, units='ft')
des_vars.add_output('OD:MNs', val=od_MNs)

mftf.connect_des_data(prob, 'DESIGN', od_pts)

for i,pt in enumerate(od_pts):
    prob.model.add_subsystem(pt, mftf.MixedFlowTurbofan(design=False))

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
prob['DESIGN.mixer.balance.P_tot']= 72.

for pt in od_pts:
    prob[pt+'.balance.FAR_core'] = 0.028
    prob[pt+'.balance.FAR_ab'] = 0.034
    prob[pt+'.balance.BPR'] = .9
    prob[pt+'.balance.W'] = 157.225
    prob[pt+'.balance.HP_Nmech'] = 1.
    prob[pt+'.balance.LP_Nmech'] = 1.
    prob[pt+'.fc.balance.Pt'] = 14.696
    prob[pt+'.fc.balance.Tt'] = 518.67
    prob[pt+'.mixer.balance.P_tot'] = 72. # 380.
    prob[pt+'.hpt.PR'] = 3.439
    prob[pt+'.lpt.PR'] = 2.438
    prob[pt+'.fan.map.RlineMap'] = 2.0
    prob[pt+'.hpc.map.RlineMap'] = 2.0

# model is very sensitive to mass flow guesses, and higher altitudes have less mass-flow
prob['OD1.balance.W'] = 80


# prob.model.DESIGN.nonlinear_solver.options['maxiter'] = 3
# prob.model.OD1.nonlinear_solver.options['maxiter'] = 0
prob.set_solver_print(level=-1)
prob.set_solver_print(level=2, depth=1)

prob.run_model()

# prob.model.DESIGN.mixer.list_outputs(residuals=True, units=True, residuals_tol=1e-3, prom_name=True)
# prob.model.DESIGN.mixer.list_outputs(residuals=True, units=True, prom_name=True)
# prob.model.DESIGN.fc.list_inputs()



mftf.page_viewer(prob, 'DESIGN')
print(3*"\n")

for pt in od_pts:
    mftf.page_viewer(prob, pt)

print()

