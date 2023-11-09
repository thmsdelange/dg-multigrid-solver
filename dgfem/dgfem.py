import numpy as np
import sympy as sym
np.set_printoptions(edgeitems=30, linewidth=100000) 
#     formatter=dict(float=lambda x: "%.3f" % x))
import scipy.sparse as sp
import scipy.integrate as si
import scipy.sparse.linalg as splin
import os

from dgfem.grid import Geometry, Grid, CoarseGrid
from dgfem.solver import Solver
from dgfem.discrete_system import DiscreteSystem
from dgfem.visualization import grid_to_vtk, elements_to_vtk
from dgfem.settings import Settings
from utils.helpers import compute_Lp_norm, compute_residual_norm, compute_row_echelon, reorder_local_to_global_DOFs, reorder_global_to_local_DOFs
from utils.logger import Logger
from utils.timer import Timer

class DGFEM:
    def __init__(self, **kwargs):
        ### build settings
        if kwargs.get('settings'):
            self.settings = kwargs.get('settings')
        else:
            from input import params
            self.settings = Settings(params)
        self.settings.update_settings(kwargs)
        self.settings.problem.exact_solution_function = self.compute_exact_solution
        self.settings.problem.MMS_source_continuity_function = self.compute_MMS_source_continuity

        self.logger = Logger(__name__, self.settings).logger
        self.timer = Timer(self.logger)

        for key, arg in kwargs.items():
            if 'solve_' in key and arg:
                self.settings.solver.method = key.removeprefix('solve_')
        self.solver = Solver(self.settings.solver.method, self.settings)
        
        ### initialize geometry and problem
        grid_filepath = os.path.join(os.getcwd(), self.settings.grid.folder, self.settings.grid.filename)
        self.geometry = Geometry(grid_filepath, self.settings)

        if self.settings.problem.type == 'Poisson':
            self.vars = ['u']
            self.P_sol = {'u': self.settings.solution.u.polynomial_degree}
            self.exact_sol = {'u': self.settings.problem.exact_solution.u}
        elif self.settings.problem.type == 'Stokes' or self.settings.problem.type == 'Navier-Stokes':
            self.vars = ['u', 'p']
            self.P_sol = {key: getattr(getattr(self.settings.solution, key), 'polynomial_degree') for key in self.vars}
            self.exact_sol = {'u': sym.sympify(self.settings.problem.exact_solution.u),
                              'v': sym.sympify(self.settings.problem.exact_solution.v),
                              'p': sym.sympify(self.settings.problem.exact_solution.p)}
            if hasattr(self.settings.problem.exact_solution, 'lam'):
                lam, nu = sym.symbols('lam nu')
                for var in self.exact_sol.keys():
                    self.exact_sol[var] = self.exact_sol.get(var).subs(lam, self.settings.problem.exact_solution.lam).subs(nu, self.settings.problem.kinematic_viscosity)
            self.exact_p_mean = self.compute_exact_pressure_mean()
        else:
            raise NotImplementedError(f"There exists no implementation for the {self.settings.problem.type} equation(s), possible equation(s) are: Poisson|Stokes|Navier-Stokes")

        ### validate settings
        self.settings._validate_settings(self.settings)

        ### building file and folder structures
        grid_filename_xyz = grid_filepath[(grid_filepath.rfind(os.sep) + 1):]
        grid_filename = grid_filename_xyz[:grid_filename_xyz.rfind('.xyz')]
        results_folder = f'exact_sol_{self.settings.problem.exact_solution.tag}'  
        if self.settings.problem.type=='Poisson':
            results_folder += f'_sigmamul{self.settings.problem.SIP_penalty_parameter_multiplier}'.replace(".", "_")
        elif self.settings.problem.type=='Stokes': 
            results_folder += f'_sigmamul{self.settings.problem.SIP_penalty_parameter_multiplier}'.replace(".", "_") + f'_gamma{self.settings.problem.velocity_penalty_parameter}'.replace(".", "_")
        self.results_dir = os.path.join('results', self.settings.problem.type.replace('-','_'), f'grid_{grid_filename}', results_folder)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.solution_visualization_filepath = os.path.join(self.results_dir, 'solution_' + '_'.join([f'P{var}{self.P_sol[var]}' for var in self.vars]))
        self.solution_summary_filepath = os.path.join(self.results_dir, 'summary.txt')

        ### initializing grids and the coefficient matrices and right-hand sides on these grids
        self.initialize()

        ### rendering vtk of the reference grid
        grid_to_vtk(os.path.join(self.results_dir, 'grid'), self.grids[0].x, self.grids[0].y)
        
        ### creating a text file to store some simulation parameters and results
        with open(self.solution_summary_filepath, 'w') as f:
            f.write("############################################\n")
            f.write("###          SIMULATION SUMMARY          ###\n")
            f.write("############################################\n\n")
            f.write(f"### grid={grid_filename}\n")
            f.write(f"### exact solution={self.exact_sol}\n")
            f.write(f"### Ni={self.geometry.Ni}, Nj={self.geometry.Nj}\n")
            f.write(f"### P grid={self.settings.grid.polynomial_degree}\n")
            f.write(f"### P sol={self.P_sol}\n")
            if self.settings.problem.type == 'Poisson':
                f.write(f"### epsilon multiplier={self.settings.problem.SIP_penalty_parameter_multiplier}\n")
            elif self.settings.problem.type == 'Stokes': 
                f.write(f"### epsilon multiplier={self.settings.problem.SIP_penalty_parameter_multiplier}\n")
                f.write(f"### gamma={self.settings.problem.velocity_penalty_parameter}\n")
            f.write("###\n")
            f.write(f"### solver={'multigrid' if self.settings.solver.method == 'multigrid' else 'direct'}\n\n")
            f.write("############################################\n\n")

    def initialize(self):
        ### initialize grid(s)
        self.sigma = self.settings.problem.SIP_penalty_parameter if self.settings.problem.SIP_penalty_parameter else (self.P_sol['u']+1)**2*self.settings.problem.SIP_penalty_parameter_multiplier  # try to build sigmas for multigrid here
        self.grids = []
        if self.settings.solver.method == 'multigrid':
            self.assemble_multigrid_operators()
        else:
            # self.grids.append(geometry.initialize(self.P_sol))
            self.grids.append(Grid(self.geometry, self.vars, self.settings.solver.discretization).initialize(self.P_sol, self.sigma))
            # self.grids[0:0] = [Grid(self.geometry, self.vars, discretization='fvm').initialize(self.P_sol, self.sigma)]
            # self.grids.append(Grid(self.geometry, self.vars, 'fvm').initialize(self.P_sol, self.sigma))
            # self.grids[0] = CoarseGrid(self.geometry, self.grids[0]).initialize(coarsening_factor=32)

        ### calculate exact solution in all grid elements on the reference grid
        reference_grid = self.grids[-1]
        self.u_exact_nodal = np.array([[self.compute_exact_solution(reference_grid.elements[i,j].x, reference_grid.elements[i,j].y, self.settings.problem.type, 'solution').get('u').get('u') for j in range(reference_grid.Nj)] for i in range(reference_grid.Ni)])
        if self.settings.problem.type=='Stokes':
            self.v_exact_nodal = np.array([[self.compute_exact_solution(reference_grid.elements[i,j].x, reference_grid.elements[i,j].y, self.settings.problem.type, 'solution').get('v').get('u') for j in range(reference_grid.Nj)] for i in range(reference_grid.Ni)])
            self.p_exact_nodal = np.array([[self.compute_exact_solution(reference_grid.elements[i,j].x, reference_grid.elements[i,j].y, self.settings.problem.type, 'solution').get('p').get('p') for j in range(reference_grid.Nj)] for i in range(reference_grid.Ni)])
            
        ### construct coefficient matrices, its splits and the right-hand sides and compute the modal solution
        discrete_system = DiscreteSystem(self.settings)
        if self.settings.solver.method == 'multigrid':
            for grid in self.grids:
                discrete_system.problem.assemble(grid)

        else:
            discrete_system.problem.assemble(reference_grid)

            if self.settings.solution.ordering == 'global': reference_grid.RHS = reorder_local_to_global_DOFs(reference_grid, reference_grid.RHS)
            if self.settings.problem.check_consistency:
                ### calculate difference between volume and surface integrals
                Epsilon = self.compute_forcing_vector_MMS(reference_grid)
                if abs(Epsilon)<1e-13: 
                    self.logger.debug("Epsilon < 1e-14, system is consistent")
                else:
                    ### calculate row echelon form(s) of continuity system
                    # np.testing.assert_allclose(reference_grid.BSR_block_D.toarray(), reference_grid.BSR_block_G.toarray().T, atol=1e-12)
                    Ainv = splin.inv(reference_grid.BSR_block_A.tocsc())
                    mat = reference_grid.BSR_block_D @ Ainv @ reference_grid.BSR_block_G
                    RHS = reference_grid.BSR_block_D @ Ainv @ reference_grid.RHS[:reference_grid.N*reference_grid.N_DOF_sol.get('u')*2] - reference_grid.RHS[-reference_grid.N*reference_grid.N_DOF_sol.get('p'):]

                    system = np.hstack([mat.toarray(), np.expand_dims(RHS, axis=1)])
                    # system = np.hstack([np.array([[1, 1], [2, 2]]),np.expand_dims(np.array([10, 21]), axis=1)])   # example of inconsistent system
                    system_ref = sym.Matrix(system).echelon_form()
                    system_ref_custom = compute_row_echelon(system)
                    system_rref = sym.Matrix(system).rref()[0]
                    self.logger.debug(f'Last part of last row of row echelon form:\n{np.array(system_ref).astype(np.float64)[-1,-3:]}')
                    self.logger.debug(f'Last part of last row of row echelon form custom:\n{system_ref_custom[-1,-3:]}')
                    self.logger.debug(f'Last part of last row of reduced row echelon form:\n{np.array(system_rref).astype(np.float64)[-1,-3:]}')
                    exit()

        self.solver.grids = self.grids

    def solve(self):
        """
        args, etc.

        Extra info: the solution is always transformed to the solution per element if needed
        """
        reference_grid = self.grids[-1]

        u_modal = self.solver.solve()

        residual_0 = compute_Lp_norm(reference_grid.RHS, 2)
        self.residual = compute_residual_norm(reference_grid, u_modal) if self.settings.solver.method != 'finite_volume_method' else compute_residual_norm(reference_grid, u_modal)
        self.logger.info(f"L2 norm of the residual (modal): {self.residual:.6e} (not normalized)")
        self.logger.info(f"L2 norm of the residual (modal): {self.residual/residual_0:.6e} (normalized)")
        if self.settings.solution.ordering == 'global': u_modal = reorder_global_to_local_DOFs(reference_grid, u_modal)
        if not self.settings.solver.method == 'finite_volume_method': u_modal = u_modal.reshape(-1,reference_grid.N_DOF_sol_tot)

        ## post processing the pressure for Stokes
        if self.settings.problem.type == 'Stokes' and self.settings.solver.method != 'smoother':
            numerical_p = 0.
            A = 0.
            k = 0
            for j in range(reference_grid.Nj):
                for i in range(reference_grid.Ni):
                    numerical_p += reference_grid.elements[i,j].compute_pressure_integral(u_modal[k,-reference_grid.N_DOF_sol.get('p'):])
                    A += reference_grid.elements[i,j].A
                    k += 1
            self.numerical_p_mean = numerical_p/A
            
            k = 0
            for j in range(reference_grid.Nj):
                for i in range(reference_grid.Ni):
                    u_modal[k,-reference_grid.N_DOF_sol.get('p')] -= 2*self.numerical_p_mean
                    k += 1
        
        ### interpolating the solution from the modes to the grid nodes
        if self.settings.solver.method != 'finite_volume_method':
            u_nodal = np.zeros((reference_grid.Ni,reference_grid.Nj,reference_grid.N_grid,reference_grid.N_grid))
            if self.settings.problem.type=='Stokes':
                v_nodal = np.zeros((reference_grid.Ni,reference_grid.Nj,reference_grid.N_grid,reference_grid.N_grid))
                p_nodal = np.zeros((reference_grid.Ni,reference_grid.Nj,reference_grid.N_grid,reference_grid.N_grid))

            if self.settings.solver.method == 'smoother_amplification':
                u_nodal = u_nodal.astype(np.complex128)
                if self.settings.problem.type=='Stokes':
                    v_nodal = v_nodal.astype(np.complex128)
                    p_nodal = p_nodal.astype(np.complex128)
            
            self.logger.debug("Interpolating the solution from modes to nodes ...")
            k = 0
            for j in range(reference_grid.Nj):
                for i in range(reference_grid.Ni):
                    u_nodal[i,j,:,:] = (reference_grid.elements[i,j].V_DOF_grid.get('u').get('u') @ u_modal[k,:reference_grid.N_DOF_sol.get('u')]).reshape(reference_grid.N_grid,reference_grid.N_grid, order='F')
                    if self.settings.problem.type=='Stokes':
                        v_nodal[i,j,:,:] = (reference_grid.elements[i,j].V_DOF_grid.get('u').get('u') @ u_modal[k,reference_grid.N_DOF_sol.get('u'):-reference_grid.N_DOF_sol.get('p')]).reshape(reference_grid.N_grid,reference_grid.N_grid, order='F')
                        p_nodal[i,j,:,:] = (reference_grid.elements[i,j].V_DOF_grid.get('p').get('u') @ u_modal[k,-reference_grid.N_DOF_sol.get('p'):]).reshape(reference_grid.N_grid,reference_grid.N_grid, order='F')
                    k += 1
        else:
            u_nodal = np.ravel(u_modal)
            v_nodal = None
            p_nodal = None

        # if self.settings.solver.method == 'smoother_amplification':
        #     A = np.abs(u_nodal)
        #     print(A[2,2,:,:])
        #     exit()
        # print(compute_Lp_norm(u_nodal, 2))

        ### calculating some error measurements
        abs_error_u = abs(u_nodal-self.u_exact_nodal)
        if self.settings.problem.type=='Stokes':
            abs_error_v = abs(v_nodal-self.v_exact_nodal)
            abs_error_p = abs(p_nodal-self.p_exact_nodal)
        self.L1_error_u = compute_Lp_norm(u_nodal-self.u_exact_nodal, 1)
        self.L2_error_u = compute_Lp_norm(u_nodal-self.u_exact_nodal, 2)
        if self.settings.problem.type=='Stokes':
            self.L1_error_v = compute_Lp_norm(v_nodal-self.v_exact_nodal, 1)
            self.L2_error_v = compute_Lp_norm(v_nodal-self.v_exact_nodal, 2)
            self.L1_error_p = compute_Lp_norm(p_nodal-self.p_exact_nodal, 1)
            self.L2_error_p = compute_Lp_norm(p_nodal-self.p_exact_nodal, 2)
        if self.settings.problem.type=='Stokes':
            self.logger.info(f"The norms of the error in u-velocity (nodal) are: L1={self.L1_error_u:.6e}, L2={self.L2_error_u:.6e}")
            self.logger.info(f"The norms of the error in v-velocity (nodal) are: L1={self.L1_error_v:.6e}, L2={self.L2_error_v:.6e}")
            self.logger.info(f"The norms of the error in pressure (nodal) are: L1={self.L1_error_p:.6e}, L2={self.L2_error_p:.6e}")
        else:
            self.logger.info(f"The norms of the error (nodal) are: L1={self.L1_error_u:.6e}, L2={self.L2_error_u:.6e}")
        
        ### exporting the solution in the elements to .vts for visualization
        if self.settings.solver.method != 'finite_volume_method' and self.settings.visualization.export:
            if self.settings.problem.type=='Stokes':
                elements_to_vtk(self.solution_visualization_filepath, reference_grid.elements, self.settings.problem.type, point_data={'u_exact': self.u_exact_nodal, 'u': u_nodal, 
                                                                                                                                        'v_exact': self.v_exact_nodal, 'v': v_nodal, 
                                                                                                                                        'pressure_exact': self.p_exact_nodal, 'pressure': p_nodal, 
                                                                                                                                        'abs_error_u': abs_error_u,
                                                                                                                                        'abs_error_v': abs_error_v,
                                                                                                                                        'abs_error_p': abs_error_p,
                                                                                                                                        })
            else:
                elements_to_vtk(self.solution_visualization_filepath, reference_grid.elements, self.settings.problem.type, point_data={'phi_exact': self.u_exact_nodal, 'phi': u_nodal, 'abs_error_phi': abs_error_u})
            
        ### exporting residual and errors and some other output
        with open(self.solution_summary_filepath, 'a') as f:
            f.write(f"Residual={self.residual}\n")
            if self.settings.problem.type=='Stokes':
                f.write(f"L1 error={self.L1_error_u} (u-velocity)\n")
                f.write(f"L2 error={self.L2_error_u} (u-velocity)\n")
                f.write(f"L1 error={self.L1_error_v} (v-velocity)\n")
                f.write(f"L2 error={self.L2_error_v} (v-velocity)\n")
                f.write(f"L1 error={self.L1_error_p} (pressure)\n")
                f.write(f"L2 error={self.L2_error_p} (pressure)\n")
            else:
                f.write(f"L1 error={self.L1_error_u}\n")
                f.write(f"L2 error={self.L2_error_u}\n")
            # f.write(f'u_modal={u_modal}')

        if self.settings.visualization.automatically_open_paraview:
            executable = self.settings.visualization.paraview_executable_path
            if not executable: raise ValueError("ParaView executable path must be set in paramfile.yml")
            import subprocess
            subprocess.Popen([rf'{executable}', self.solution_visualization_filepath+'.vts'])
        
    
    def assemble_multigrid_operators(self):
        """Keep in mind that the multigrid cycle runs from the highest level k, to k-1, k-2, k-3, to 0""" 
        self.solver.restriction_operators = []
        self.solver.prolongation_operators = []
        self.solver.multigrid_type = []

        if self.settings.solver.multigrid.penalty_parameter_coarsening.enabled:
            sigmas = []
            sigma_min = (self.P_sol.get('u')+1)**2
            sigma_multipliers = sorted(list(map(int, self.settings.solver.multigrid.penalty_parameter_coarsening.multipliers.split(','))))
            for multiplier in sigma_multipliers:
                sigmas.append(sigma_min*multiplier)
                if multiplier < 2:
                    self.logger.warning("You are trying to use a penalty parameter multiplier lower than 2, expect unstable results on curved grids")

            self.grids[0:0] = [Grid(self.geometry, self.vars).initialize(self.P_sol, sigma) for sigma in sigmas]   # insert list of grids at the beginning of self.grids list
            restriction_operator = np.eye((self.P_sol.get('u')+1)**2)
            prolongation_operator = restriction_operator.T
            self.solver.restriction_operators[0:0] = [restriction_operator for _ in range(len(sigmas) -1)]
            self.solver.prolongation_operators[0:0] = [prolongation_operator for _ in range(len(sigmas) -1)]
            self.solver.multigrid_type[0:0] = ['penalty_parameter' for _ in range(len(sigmas) -1)]
        if self.settings.solver.multigrid.polynomial_coarsening.enabled:
            p_levels = {var: sorted(list(map(int, getattr(self.settings.solver.multigrid.polynomial_coarsening.levels, var).split(',')))) for var in self.vars}
            if self.settings.solver.multigrid.penalty_parameter_coarsening.enabled:
                p_levels_grids = {var: p_levels.get(var)[:-1] for var in self.vars}
                self.settings.problem.SIP_penalty_parameter_multiplier = sigma_multipliers[0]
            else:
                p_levels_grids = p_levels

            sigma_min = [(p_sol_i+1)**2*self.settings.problem.SIP_penalty_parameter_multiplier for p_sol_i in p_levels_grids.get('u')]

            ### initialize grids (needed to construct coarse grid operators)
            self.grids[0:0] = [Grid(self.geometry, self.vars).initialize(dict(zip(p_levels_grids.keys(), p_sol_i)), sigma_min_i) for p_sol_i, sigma_min_i in zip(zip(*p_levels_grids.values()), sigma_min)] # insert list of grids at the beginning of self.grids list

            ### construct restriction operators
            p_restriction_operators = []
            p_prolongation_operators = []
            for i, level in enumerate(p_levels.get('u')[:-1]):
                p_fine = p_levels.get('u')[i+1]
                p_coarse = level
                N_fine = (p_fine+1)**2
                N_coarse = (p_coarse+1)**2

                restriction_operator = np.eye(N_coarse)
                for i in range(p_coarse):
                    restriction_operator = np.insert(restriction_operator, (i+1)*(p_coarse+1)+i*(p_fine-p_coarse), np.zeros((p_fine-p_coarse,N_coarse)), axis=1)
                restriction_operator = np.append(restriction_operator, np.zeros((N_coarse,N_fine-N_coarse-(p_fine-p_coarse)*p_coarse)), axis=1)
                p_restriction_operators.append(restriction_operator)
                p_prolongation_operators.append(restriction_operator.T)


                self.solver.restriction_operators[0:0] = p_restriction_operators
                self.solver.prolongation_operators[0:0] = p_prolongation_operators
            self.solver.multigrid_type[0:0] = ['polynomial' for _ in range(len(p_levels.get('u')) -1)]
        if self.settings.solver.multigrid.geometric_coarsening.enabled:
            if not self.grids: self.grids.append(Grid(self.geometry, self.vars).initialize(self.P_sol, self.sigma))
            if self.settings.solver.multigrid.geometric_coarsening.use_FVM:
                sigma_min = 1.*self.settings.problem.SIP_penalty_parameter_multiplier
                self.grids[0:0] = [Grid(self.geometry, self.vars, discretization='fvm').initialize(self.P_sol, self.sigma)]
                restriction_operator = np.array([[1., 0., 0., 0.]])
                prolongation_operator = restriction_operator.T
                self.solver.restriction_operators[0:0] = [restriction_operator/2.]
                self.solver.prolongation_operators[0:0] = [prolongation_operator*2.]
                self.solver.multigrid_type[0:0] = ['geometric']

            # if not self.settings.solver.multigrid.geometric_coarsening.use_FVM:
            if True:
                coarsening_factors = self.settings.solver.multigrid.geometric_coarsening.coarsening_factors
                coarsening_factors = sorted(list(map(int, coarsening_factors.split(','))), reverse=True) if not isinstance(coarsening_factors, int) else [coarsening_factors]
                
                if self.settings.solver.multigrid.geometric_coarsening.use_FVM:
                    self.grids[0:0] = [CoarseGrid(self.geometry, self.grids[0], self.vars, discretization='fvm').initialize(coarsening_factor=coarsening_factor) for coarsening_factor in coarsening_factors]
                    # if len(coarsening_factors)>1:
                    prolongation_operator = np.array([np.array([9., 0., 0., 0.])/16.,
                                                    np.array([9., 3., 0., 0.])/16.,
                                                    np.array([3., 9., 0., 0.])/16.,
                                                    np.array([0., 9., 0., 0.])/16.,
                                                    np.array([9., 0., 9., 0.])/16.,
                                                    np.array([9., 3., 3., 1.])/16.,
                                                    np.array([3., 9., 1., 3.])/16.,
                                                    np.array([0., 9., 0., 3.])/16.,
                                                    np.array([3., 0., 9., 0.])/16.,
                                                    np.array([3., 1., 9., 3.])/16.,
                                                    np.array([1., 3., 3., 9.])/16.,
                                                    np.array([0., 3., 0., 9.])/16.,
                                                    np.array([0., 0., 9., 0.])/16.,
                                                    np.array([0., 0., 9., 3.])/16.,
                                                    np.array([0., 0., 3., 9.])/16.,
                                                    np.array([0., 0., 0., 9.])/16.,
                                                    ])
                    restriction_operator = prolongation_operator.T/4.
                    # restriction_operator *= 4.
                    # prolongation_operator *= 4.
                else:
                    self.grids[0:0] = [CoarseGrid(self.geometry, self.grids[0], self.vars).initialize(coarsening_factor=coarsening_factor) for coarsening_factor in coarsening_factors]
                    restriction_operator = np.array([np.array([1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.])/4.,
                                                    np.array([-np.sqrt(3), 1., 0., 0., np.sqrt(3), 1., 0., 0., -np.sqrt(3), 1., 0., 0., np.sqrt(3), 1., 0., 0.,])/8.,
                                                    np.array([-np.sqrt(3), 0., 1., 0., -np.sqrt(3), 0., 1., 0., np.sqrt(3), 0., 1., 0., np.sqrt(3), 0., 1., 0.,])/8.,
                                                    np.array([3., -np.sqrt(3), -np.sqrt(3), 1., -3., -np.sqrt(3), np.sqrt(3), 1., -3., np.sqrt(3), -np.sqrt(3), 1., 3., np.sqrt(3), np.sqrt(3), 1.,])/16.
                                                    ])
                    prolongation_operator = restriction_operator.T*4.

                # coarsening_factors_range = range(len(coarsening_factors)) if not self.settings.solver.multigrid.geometric_coarsening.use_FVM else range(len(coarsening_factors)-1)
                self.solver.restriction_operators[0:0] = [restriction_operator for _ in range(len(coarsening_factors))]
                self.solver.prolongation_operators[0:0] = [prolongation_operator for _ in range(len(coarsening_factors))]

                self.solver.multigrid_type[0:0] = ['geometric' for _ in range(len(coarsening_factors))]
        
        self.logger.debug("Multigrid levels:")
        for idx, grid in enumerate(self.grids):
            self.logger.debug(f'grid number {idx+1}: P_grid={grid.P_grid}, P_sol={grid.P_sol}, sigma={grid.sigma}, Ni={grid.Ni}, Nj={grid.Nj}')
        # for idx, restriction_operator in enumerate(self.solver.restriction_operators):
        #     print(f'restriction operator number {idx+1}: restriction_operator.shape=\n{restriction_operator.shape}')
        #     # print(f'restriction operator number {idx+1}: restriction_operator=\n{restriction_operator}')
        # exit()
        
  
    def compute_exact_pressure_mean(self):
        x, y, r, theta = sym.symbols('x y r theta')
        p_exact = self.exact_sol.get('p')
        if self.settings.grid.circular:
            r_min, r_max = np.min(self.geometry.x[0,:]), np.max(self.geometry.x[0,:])
            theta_min, theta_max = 0, 2*np.pi
            
            A = sym.integrate(r, (r, (r_min, r_max)), (theta, (theta_min, theta_max)))
            if (x in p_exact.free_symbols or y in p_exact.free_symbols) or ((not r in p_exact.free_symbols and not theta in p_exact.free_symbols) and not isinstance(p_exact, sym.Number)): 
                p_exact = p_exact.subs(x, r*sym.cos(theta)).subs(y, r*sym.sin(theta)) #.rewrite(sym.exp).expand()
                f = sym.lambdify((r, theta), p_exact, "numpy")
                p, _ = si.dblquad(f, r_min, r_max, theta_min, theta_max, epsabs=1e-12, epsrel=1e-16)
                p_mean = p/A
            else:
                p_mean = sym.integrate(p_exact*r, (r, (r_min, r_max)), (theta, (theta_min, theta_max)))/A
        else:
            if (r in p_exact.free_symbols or theta in p_exact.free_symbols) or ((not x in p_exact.free_symbols and not y in p_exact.free_symbols) and not isinstance(p_exact, sym.Number)):
                raise ValueError("The pressure must be defined in terms of x and y on a rectangular grid")
            
            x_min, x_max = np.min(self.geometry.x), np.max(self.geometry.x)
            y_min, y_max = np.min(self.geometry.y), np.max(self.geometry.y)
            
            A = sym.integrate(1, (x, (x_min, x_max)), (y, (y_min, y_max)))
            p_mean = sym.integrate(p_exact, (x, (x_min, x_max)), (y, (y_min, y_max)))/A
        return float(p_mean)
    
    def compute_MMS_source_continuity(self, xi, yi, grid):
        f_cont = self.compute_exact_solution(xi, yi, 'Stokes', 'source_continuity')
        for var, f_cont_var in f_cont.items():
            f_cont[var] = f_cont_var + grid.Epsilon
        return f_cont
    
    def compute_exact_solution(self, xi, yi, problem, which='solution'):
        """
        if which='solution' and problem='Poisson', u is returned
        if which='solution' and problem='Stokes', u,v,p is returned

        g_int should compute u in u and p integration nodes etc.
        """
        x, y, r, theta = sym.symbols('x y r theta')
        if problem=='Poisson':
            sol = {'u': self.exact_sol.get('u')}
        elif problem=='Stokes':
            sol = {key: self.exact_sol.get(key) for key in self.exact_sol.keys()}
            # if self.geometry.O_grid:
            #     sol['p'] = sol['p'].subs(r, sym.sqrt(x**2+y**2)).subs(theta, sym.atan(y/x))
            
            f_cont = sym.diff(sol.get('u'), x) + sym.diff(sol.get('v'), y)
            if self.settings.solution.manufactured_solution:
                if not f_cont.is_zero:
                    self.logger.error(f"Manufactured solution is not divergence-free, {f_cont=}")
                    raise ValueError("Manufactured solution is not divergence-free")

        if which=='solution':
            sol_lambdas = {}
            for var_sol in sol.keys():
                result = {}
                # print(f'{sol.get(var_sol)=}')
                for var in self.vars:
                    xii = xi.get(var) if isinstance(xi, dict) else xi
                    yii = yi.get(var) if isinstance(yi, dict) else yi
                    if isinstance(sol.get(var_sol), sym.Number):
                        assert xii.shape == yii.shape
                        sol_lambda = lambda xii, yii: np.full_like(xii, sol.get(var_sol))
                    else:
                        sol_lambda = sym.lambdify((x, y), sol.get(var_sol))
                    result[var] = np.squeeze(sol_lambda(xii, yii) - self.exact_p_mean) if var_sol=='p' else np.squeeze(sol_lambda(xii, yii))
                    if result[var].shape == (): result[var] = np.array([result[var]])
                sol_lambdas[var_sol] = result
            return sol_lambdas
        elif which=='source_continuity':
            source_lambdas = {}
            for var in self.vars:
                # print(f"{var}: {f}")
                if isinstance(xi, dict): xii = xi.get(var)
                if isinstance(yi, dict): yii = yi.get(var)
                if isinstance(f_cont, sym.Number):
                    assert xii.shape == yii.shape
                    source_lambda = lambda xii, yii: np.full_like(xii, f_cont)
                else:
                    source_lambda = sym.lambdify((x,y), f_cont)
                source = source_lambda(xii, yii)
                source_lambdas[var] = np.squeeze(source) if source.shape[0]>1 and source.shape[1]>1 else source
            return source_lambdas
        elif which=='source_momentum':
            grad_u = [self.settings.problem.kinematic_viscosity*sym.diff(sol.get('u'), x), self.settings.problem.kinematic_viscosity*sym.diff(sol.get('u'), y)]
            laplace_u = -(sym.diff(grad_u[0], x) + sym.diff(grad_u[1], y))     # divergence of the gradient of u
            if problem=='Poisson':
                f_vec = [laplace_u]
            elif problem=='Stokes':
                grad_v = [self.settings.problem.kinematic_viscosity*sym.diff(sol.get('v'), x), self.settings.problem.kinematic_viscosity*sym.diff(sol.get('v'), y)]
                laplace_v = -(sym.diff(grad_v[0], x) + sym.diff(grad_v[1], y))     # divergence of the gradient of u
                grad_p = [sym.diff(sol.get('p'), x), sym.diff(sol.get('p'), y)]
                f_vec = [laplace_u + grad_p[0], laplace_v + grad_p[1]]
            
            source_lambdas = []
            for f in f_vec:
                result = {}
                for var in self.vars:
                    if isinstance(xi, dict): xii = xi.get(var)
                    if isinstance(yi, dict): yii = yi.get(var)
                    if isinstance(f, sym.Number):
                        assert xii.shape == yii.shape
                        source_lambda = lambda xii, yii: np.full_like(xii, f)
                    else:
                        source_lambda = sym.lambdify((x,y), f)
                    result[var] = np.squeeze(source_lambda(xii, yii))
                source_lambdas.append(result)
            return source_lambdas