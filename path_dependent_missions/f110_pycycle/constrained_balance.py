from openmdao.api import ImplicitComponent

class ConstrainedTempBalance(ImplicitComponent):

    def setup(self):

        self.add_input('T_computed', units='degR')
        self.add_input('T_requested', units='degR')

        self.add_input('Nc_computed', units='rpm')
        self.add_input('Nc_limit', units='rpm', val=1.05)

        self.add_output('FAR')

        self.declare_partials('FAR', ['T_computed', 'T_requested', 'Nc_computed', 'Nc_limit'])

        self.set_check_partial_options(wrt="*", method='cs')

    def apply_nonlinear(self, inputs, outputs, residuals):

        if inputs['Nc_computed'] < inputs['Nc_limit']:
            residuals['FAR'] = inputs['T_computed']/inputs['T_requested'] - 1

        else:
            residuals['FAR'] = inputs['Nc_computed']/inputs['Nc_limit'] - 1

    def linearize(self, inputs, outputs, J):

        T_req = inputs['T_requested']
        T_com = inputs['T_computed']

        Nc_com = inputs['Nc_computed']
        Nc_lim = inputs['Nc_limit']

        if inputs['Nc_computed'] < inputs['Nc_limit']:
            J['FAR', 'T_computed'] = 1/T_req
            J['FAR', 'T_requested'] = -T_com/T_req**2

            J['FAR', 'Nc_computed'] = 0.
            J['FAR', 'Nc_limit'] = 0.

        else:
            J['FAR', 'T_computed'] = 0.
            J['FAR', 'T_requested'] = 0.

            J['FAR', 'Nc_computed'] = 1/Nc_lim
            J['FAR', 'Nc_limit'] = -Nc_com/Nc_lim**2


if __name__ == "__main__":

    from openmdao.api import Problem

    p = Problem()
    p.model = ConstrainedTempBalance()

    p.setup(force_alloc_complex=True)

    p['Nc_limit'] = 1
    p['Nc_computed'] = .9
    p['T_computed'] = 3400.
    p['T_requested'] = 3200.
    p.check_partials()

    print('\n', 5*'#####', '\n')


    p['Nc_limit'] = 1
    p['Nc_computed'] = 1.1
    p['T_computed'] = 3400.
    p['T_requested'] = 3200.
    p.check_partials()
