from __future__ import print_function
from six import iteritems
from openmdao.api import CaseReader

def read_db(filename):

    dvs = {
        'r' : 'climb.states:r',
        'h' : 'climb.states:h',
        'v' : 'climb.states:v',
        'gam' : 'climb.states:gam',
        'm' : 'climb.states:m',
        'alpha' : 'climb.controls:alpha',
        't_duration' : 'climb.t_duration',
    }

    states = {
    }

    cons = {
    }

    objs = {
    }

    data_all_iters = []

    cr = CaseReader(filename)

    case_keys = cr.driver_cases.list_cases()
    for case_key in case_keys:

        case = cr.driver_cases.get_case(case_key)

        data = {}

        for key, cr_key in iteritems(dvs):
            data[key] = case.desvars[cr_key]

        for key, cr_key in iteritems(states):
            data[key] = case.sysincludes[cr_key]

        for key, cr_key in iteritems(objs):
            data[key] = case.objectives[cr_key]

        for key, cr_key in iteritems(cons):
            data[key] = case.constraints[cr_key]

        data_all_iters.append(data)

    return data_all_iters
