from __future__ import print_function, division, absolute_import

from path_dependent_missions.simple_heat.simple_heat_problem import setup_energy_opt, plot_results

q_tank = 10.
q_hx1 = 0.
q_hx2 = -20.
num_seg = 5
order = 5

p = setup_energy_opt(num_seg, order, q_tank, q_hx1, q_hx2, opt_burn=False)

p.run_driver()
plot_results(p)
