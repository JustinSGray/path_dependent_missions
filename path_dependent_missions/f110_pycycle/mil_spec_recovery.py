from openmdao.api import ExplicitComponent

class MilSpecRecovery(ExplicitComponent):

    def setup(self):

        self.add_input('MN', desc='Mach Number', val=0.5)
        self.add_output('ram_recovery', desc='percent total pressure recovered')

        self.declare_partials('ram_recovery', 'MN')

    def compute(self, inputs, outputs):

        MN = inputs['MN']
        if MN <= 1.:
            outputs['ram_recovery'] = 1.
        elif (MN > 1.) and (MN <= 5.):
            outputs['ram_recovery'] = 1. - 0.076*(MN - 1.)**1.35
        else:
            outputs['ram_recovery'] = 800/(MN**4 + 935)

    def compute_partials(self, inputs, J):
        MN = inputs['MN']

        if MN <= 1.:
            J['ram_recovery', 'MN'] = 0.
        elif (MN > 1.) and (MN <= 5.):
            J['ram_recovery', 'MN'] = -0.76*1.35*(MN - 1)**0.35
        else:
            J['ram_recovery', 'MN'] = -800.*4*MN**3/(MN**4 + 935.)**2


if __name__ == "__main__":
    from openmdao.api import Problem

    p = Problem()
    p.model = MilSpecRecovery()

    p.setup()

    # note: derivatives are discontinuous at MN = 1 and MN = 5
    # check partials will have trouble at these points
    # make sure to check partials in all three ranges
    print(80*'#')
    p['MN'] = 0.5
    p.check_partials()

    print(80*'#')
    p['MN'] = 1.2
    p.check_partials()

    print(80*'#')
    p['MN'] = 6.0
    p.check_partials()
