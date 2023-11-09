from input import params    # ideally Settings is built from input params

class Settings:
    def __init__(self, params) -> None:
        self.settings = self._load_settings(params)

    def _load_settings(self, params):
        for key, value in params.items():
            if isinstance(value, dict):
                setattr(self, key.replace(' ', '_'), Settings(value))
            else:
                setattr(self, key.replace(' ', '_'), value)

    def _attribute_exists(self, attribute_path):
        keys = attribute_path.split('.')
        obj = self
        for key in keys:
            if not hasattr(obj, key):
                return False
            obj = getattr(obj, key)
        return True
    
    def _validate_settings(self, settings):
        if settings.solver.method == 'smoother_amplification':
            assert settings.problem.type == 'Poisson'
            if settings.solver.discretization == 'dg':
                assert settings.solution.u.polynomial_degree == 6
            elif settings.solver.discretization == 'fvm':
                assert settings.solution.u.polynomial_degree == 0

        if settings.problem.type == 'Poisson':
            assert settings.solution.ordering == 'local'
        if settings.problem.type == 'Stokes':
            if settings.solver.method == 'multigrid':
                assert settings.solution.ordering == 'global'
                assert settings.problem.multiply_inverse_mass_matrix == True

    def update_setting(self, attribute_path, new_value):
        if not self._attribute_exists: raise AttributeError(f'Attribute "{attribute_path}" does not exist!')
        keys = attribute_path.split('.')
        obj = self
        for key in keys[:-1]:
            obj = getattr(obj, key)
        setattr(obj, keys[-1], new_value)
    
    def update_settings(self, kwargs):
        if kwargs.get('grid_folder'): self.update_setting('grid.folder', kwargs.get('grid_folder'))
        if kwargs.get('grid_file'): self.update_setting('grid.filename', kwargs.get('grid_file'))
        if kwargs.get('p_grid'): self.update_setting('grid.polynomial_degree', kwargs.get('p_grid'))
        if kwargs.get('p_solution'): self.update_setting('solution.polynomial_degree', kwargs.get('p_solution'))
        if kwargs.get('check_eigenvalues'): self.update_setting('solver.check_eigenvalues', True)
        if kwargs.get('check_condition_number'): self.update_setting('problem.check_condition_number', True)
        if kwargs.get('plot_sparsity_pattern'): self.update_setting('visualization.plot_sparsity_pattern', True)
        
        if kwargs.get('manufactured_solution'): self.update_setting('solution.manufactured_solution', kwargs.get('manufactured_solution'))
        if kwargs.get('solution_polynomial_degree_u'): self.update_setting('solution.u.polynomial_degree', kwargs.get('solution_polynomial_degree_u'))
        if kwargs.get('solution_polynomial_degree_p'): self.update_setting('solution.p.polynomial_degree', kwargs.get('solution_polynomial_degree_p'))
        if kwargs.get('solution_ordering'): self.update_setting('solution.ordering', kwargs.get('solution_ordering'))
        if kwargs.get('problem_governing_equations'): self.update_setting('problem.governing_equation(s)', kwargs.get('problem_governing_equations'))
        if kwargs.get('problem_kinematic_viscosity'): self.update_setting('problem.kinematic_viscosity', kwargs.get('problem_kinematic_viscosity'))
        if kwargs.get('SIP_penalty_parameter'): self.update_setting('problem.SIP_penalty_parameter', kwargs.get('SIP_penalty_parameter'))
        if kwargs.get('SIP_penalty_parameter_multiplier'): self.update_setting('problem.SIP_penalty_parameter multiplier', kwargs.get('SIP_penalty_parameter_multiplier'))
        if kwargs.get('velocity_penalty_parameter'): self.update_setting('problem.velocity_penalty_parameter', kwargs.get('velocity_penalty_parameter'))
        if kwargs.get('exact_solution_u'): self.update_setting('problem.exact_solution.u', kwargs.get('exact_solution_u'))
        if kwargs.get('exact_solution_v'): self.update_setting('problem.exact_solution.v', kwargs.get('exact_solution_v'))
        if kwargs.get('exact_solution_p'): self.update_setting('problem.exact_solution.p', kwargs.get('exact_solution_p'))
        if kwargs.get('exact_solution_tag'): self.update_setting('problem.exact_solution.tag', kwargs.get('exact_solution_tag'))
        if kwargs.get('smoother'): self.update_setting('solver.smoother', kwargs.get('smoother'))
        if kwargs.get('discretization'): 
            self.update_setting('solver.discretization', kwargs.get('discretization')) 
        else: 
            self.update_setting('solver.discretization', 'dg')
        if kwargs.get('solve_finite_volume_method'): self.update_setting('solver.discretization', 'fvm')