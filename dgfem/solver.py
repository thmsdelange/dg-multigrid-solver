import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
import pyamg
import os
import pickle

from dgfem.relaxation import Relaxation
from utils.helpers import compute_Lp_norm
from utils.logger import Logger
from utils.timer import Timer

class Solver:
    def __init__(self, method, settings) -> None:
        self.settings = settings
        self.logger = Logger(__name__, self.settings).logger
        self.timer = Timer(self.logger)
        self.method = method
        self.grids = []
        self.restriction_operators = []
        self.prolongation_operators = []
        self.multigrid_type = []
        self.residuals = []

    def solve(self):
        self.logger.debug(f"Solving with {self.method} method ...")
        reference_grid = self.grids[-1]
        if self.method == 'direct':
            with Timer() as timer:
                u = self.solve_directly(reference_grid, reference_grid.RHS)
        elif self.method == 'smoother':
            with Timer() as timer:
                u = self.solve_smoother(reference_grid, reference_grid.RHS)
        elif self.method == 'smoother_amplification':
            with Timer() as timer:
                u = Relaxation.calculate_amplification(reference_grid)
                # u = Relaxation.calculate_amplification(reference_grid, 'finite_volume_method')
        elif self.method == 'pyamg':
            with Timer() as timer:
                u = self.solve_pyamg(reference_grid, reference_grid.RHS)
        elif self.method == 'krylov':
            with Timer() as timer:
                u = self.solve_krylov(reference_grid, reference_grid.RHS)
        elif self.method == 'multigrid':
            RHS_0 = reference_grid.RHS
            u_0 = np.zeros_like(RHS_0)
            k_0 = len(self.grids)
            with Timer() as timer:
                u = self.solve_multigrid(levels=k_0, RHS=RHS_0, u=u_0, tol=self.settings.solver.multigrid.tolerance, max_cycles=self.settings.solver.multigrid.max_cycles)
        elif self.method == 'finite_volume_method':
            with Timer() as timer:
                u = splin.spsolve(reference_grid.BSR, reference_grid.RHS)
        self.logger.info(f"Solving with {self.method} method took {timer.elapsed():.4g} seconds")
        return u

    def solve_directly(self, grid, RHS):
        self.logger.info("Solving the system of equations directly ...")
        u = splin.spsolve(grid.BSR.tocsr(), RHS)
        return u

    def solve_smoother(self, grid, RHS):
        if self.settings.solver.smoother == 'distributive_gauss_seidel':
            u = Relaxation.distributive_gauss_seidel(grid, RHS, max_iterations=1000000, splitting='lsq', settings=self.settings)
        else:
            u = getattr(Relaxation, self.settings.solver.smoother)(grid, RHS, max_iterations=100, direction='symmetric')
        return u
    
    def solve_pyamg(self, grid, RHS):
        # ml = pyamg.ruge_stuben_solver(sp.csr_matrix(grid.BSR), presmoother='block_gauss_seidel', postsmoother='block_gauss_seidel')
        ml = pyamg.ruge_stuben_solver(sp.csr_matrix(grid.BSR.tocsr()))
        residuals = []
        u, info = ml.solve(RHS, tol=1e-6, maxiter=1000, residuals=residuals, return_info=True)
        print('residuals')
        for i, res in enumerate(residuals):
            print(f'Residual at iteration {i}:\t{res:.6e}')
        print(f'Info code: {info}')
        return u
    
    def solve_krylov(self, grid, RHS):  # broken
        ### this function is currently broken
        
        # from scipy.sparse.linalg import spilu, LinearOperator, gmres, spsolve
        # M = LinearOperator(matvec=spilu(grid.BSR.tocsc(), fill_factor=20, drop_rule='dynamic').solve,
        #                     shape=grid.BSR.toarray().shape, 
        #                     dtype=np.float64)
        # P_iLU = splin.spilu(grid.BSR.tocsc())
        # M = splin.LinearOperator(grid.BSR.shape, P_iLU.solve)

        if not isinstance(grid.BSR_block_A_D, sp.bsr_array): _, grid.BSR_block_A_D, _ = Relaxation.split_block_EDF(grid.BSR_block_A)
        ### approximate inv(A) with the inverse of its block diagonal
        Ainv = splin.inv(grid.BSR_block_A_D.tocsc())
        S = -grid.BSR_block_D @ Ainv @ grid.BSR_block_G
        # Sinv = splin.inv(S.tocsc())
        P_d = sp.bsr_array(sp.vstack([sp.hstack([grid.BSR_block_A, sp.bsr_array(grid.BSR_block_G.shape, dtype=grid.BSR_block_G.dtype)]), 
                                      sp.hstack([sp.bsr_array(grid.BSR_block_D.shape, dtype=grid.BSR_block_D.dtype) , -S])], format='bsr'))
        # M_t1 = sp.bsr_array(sp.vstack([sp.hstack([Ainv, sp.bsr_array(grid.BSR_block_G.shape, dtype=grid.BSR_block_G.dtype)]), 
        #                                sp.hstack([sp.bsr_array(grid.BSR_block_D.shape, dtype=grid.BSR_block_D.dtype) , sp.bsr_array(sp.identity(S.shape[0], format='bsr'))])], format='bsr'))
        # M_t2 = sp.bsr_array(sp.vstack([sp.hstack([sp.bsr_array(sp.identity(Ainv.shape[0], format='bsr')), grid.BSR_block_G]), 
        #                                sp.hstack([sp.bsr_array(grid.BSR_block_D.shape, dtype=grid.BSR_block_D.dtype) , -sp.bsr_array(sp.identity(S.shape[0], format='bsr'))])], format='bsr'))
        # M_t3 = sp.bsr_array(sp.vstack([sp.hstack([sp.bsr_array(sp.identity(Ainv.shape[0], format='bsr')), sp.bsr_array(grid.BSR_block_G.shape, dtype=grid.BSR_block_G.dtype)]), 
        #                                sp.hstack([sp.bsr_array(grid.BSR_block_D.shape, dtype=grid.BSR_block_D.dtype) , Sinv])], format='bsr'))
        
        M_d = splin.inv(P_d.tocsc())
        # M_t = M_t1 @ M_t2 @ M_t3
        # M_u = lambda u: splin.spsolve(P_d.tocsc(), u)
        # M = splin.LinearOperator(P_d.shape, M_u)
        
        u, info = splin.lgmres(grid.BSR, RHS, M=M_d, atol=1e-5)
        self.logger.info(f"Krylov solver exited after {info} iterations")
        
        # u, info = gmres(grid.BSR, RHS, tol=1e-16)
        return u
    
    def solve_multigrid(self, levels:int, RHS:np.ndarray, u:np.ndarray, tol=1e-6, max_cycles=100):
        ### performing multigrid cycles
        n = 0
        residual_0 = compute_Lp_norm(self.grids[-1].RHS - self.grids[-1].BSR @ u, 2)
        while n<max_cycles:
            residual = compute_Lp_norm(self.grids[-1].RHS - self.grids[-1].BSR @ u, 2)/residual_0
            self.logger.debug(f"The L2 norm of the normalised residual (modal): {residual:.6e}")
            self.residuals.append(residual)
            if residual<tol or np.isnan(residual) or np.isinf(residual):
                break
            self.logger.debug(f"Performing V cycle {n+1} ...")
            u = self.multigrid_V_cycle(k=levels, RHS=RHS, u=u)
            n += 1

        residuals_path = os.path.join(os.getcwd(), 'postprocessing', 'pickles', 'multigrid')
        if not os.path.exists(residuals_path):
            os.makedirs(residuals_path)
        grid = self.grids[-1]
        residuals_file = f'residuals_{self.settings.problem.type}_{grid.Ni}X{grid.Nj}_nPoly{grid.P_grid}'
        residuals_file += '_' + '_'.join([multigrid_type for multigrid_type in sorted(set(self.multigrid_type))])
        residuals_file += '_circle' if self.settings.grid.circular else '_rectangle'
        residuals_file += '.pkl'
        pickle_path = os.path.join(residuals_path, residuals_file)
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.residuals, f)
        return u

    def multigrid_V_cycle(self, k, RHS, u):
        if k>1:
            pre_smoother = getattr(getattr(self.settings.solver.multigrid, f'{self.multigrid_type[k-2]}_coarsening'), 'pre_smoother')
            post_smoother = getattr(getattr(self.settings.solver.multigrid, f'{self.multigrid_type[k-2]}_coarsening'), 'post_smoother')
            
            ### perform nu_1 pre-relaxation sweeps
            u = getattr(Relaxation, pre_smoother.smoother)(grid=self.grids[k-1], RHS=RHS, u=u, direction=pre_smoother.direction, max_iterations=pre_smoother.iterations, omega=pre_smoother.relaxation_factor)

            ### compute residual and restrict to coarse grid
            residual = RHS - self.grids[k-1].BSR @ u

            if self.multigrid_type[k-2]=='geometric':
            # if self.multigrid_type[k-2]=='geometric' and not self.settings.solver.multigrid.geometric_coarsening.use_FVM:
                if not self.settings.solver.multigrid.geometric_coarsening.use_FVM:
                    N_DOFs = self.grids[k-2].N_DOF_sol.get('u')  
                    coarsening_factor = 2
                    Ni_el = int(self.grids[k-2].Ni)
                    Nj_el = int(self.grids[k-2].Nj)
                else:
                   N_DOFs = 1
                   coarsening_factor = 4
                   Ni_el = int(self.grids[k-2].Ni/2)
                   Nj_el = int(self.grids[k-2].Nj/2)
                residual = residual.reshape((Ni_el, coarsening_factor, Nj_el, coarsening_factor, N_DOFs)).transpose((0, 2, 1, 3, 4))
                
            residual = residual.reshape((-1,self.restriction_operators[k-2].shape[1]))
            residual_coarse = np.einsum('ij,kj->ki', self.restriction_operators[k-2], residual)
            RHS_coarse = np.ravel(residual_coarse)  # not order='F' because by definition of the restriction operator, residual coarse is already sorted correctly

            ### coarsen u to level k-1
            u_coarse = self.multigrid_V_cycle(k=k-1, RHS=RHS_coarse, u=np.zeros_like(RHS_coarse))
            
            ### Interpolate the grid to k
            u_coarse = u_coarse.reshape((-1,self.prolongation_operators[k-2].shape[1]))
            v = np.einsum('ij,kj->ki', self.prolongation_operators[k-2], u_coarse)

            if self.multigrid_type[k-2]=='geometric':
            # if self.multigrid_type[k-2]=='geometric' and not self.settings.solver.multigrid.geometric_coarsening.use_FVM:
                N_DOFs = self.grids[k-2].N_DOF_sol.get('u') if not self.settings.solver.multigrid.geometric_coarsening.use_FVM else 1
                if not self.settings.solver.multigrid.geometric_coarsening.use_FVM:
                    N_DOFs = self.grids[k-2].N_DOF_sol.get('u')  
                    coarsening_factor = 2
                    Ni_el = int(self.grids[k-2].Ni)
                    Nj_el = int(self.grids[k-2].Nj)
                else:
                   N_DOFs = 1
                   coarsening_factor = 4
                   Ni_el = int(self.grids[k-2].Ni/2)
                   Nj_el = int(self.grids[k-2].Nj/2)
                v = v.reshape((Ni_el, Nj_el, coarsening_factor, coarsening_factor, N_DOFs)).transpose((0, 2, 1, 3, 4))

            ### perform correction
            u += np.ravel(v)
            
            ### perform nu_2 post-relaxation sweeps
            u = getattr(Relaxation, post_smoother.smoother)(grid=self.grids[k-1], RHS=RHS, u=u, direction=post_smoother.direction, max_iterations=post_smoother.iterations, omega=post_smoother.relaxation_factor)
        else:
            ### solving the error on the coarsest level
            if self.settings.solver.multigrid.coarse_grid_solver == 'direct':
                u = self.solve_directly(self.grids[k-1], RHS)
            elif self.settings.solver.multigrid.coarse_grid_solver == 'smoother':
                pre_smoother = getattr(getattr(self.settings.solver.multigrid, f'{self.multigrid_type[k-1]}_coarsening'), 'pre_smoother')
                post_smoother = getattr(getattr(self.settings.solver.multigrid, f'{self.multigrid_type[k-1]}_coarsening'), 'post_smoother')
                u = getattr(Relaxation, pre_smoother.smoother)(grid=self.grids[k-1], RHS=RHS, u=u, direction=pre_smoother.direction, max_iterations=10, omega=pre_smoother.relaxation_factor)
            elif self.settings.solver.multigrid.coarse_grid_solver == 'amg':
                u = pyamg.ruge_stuben_solver(sp.csr_matrix(self.grids[k-1].BSR)).solve(RHS, tol=1e-2)
        return u

