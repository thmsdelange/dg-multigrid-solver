from itertools import chain
import numpy as np
import scipy as sci
import scipy.sparse as sp
import scipy.sparse.linalg as splin
import pyamg.amg_core as amgc
import os
import pickle

import dgfem.pyamg_relaxation as amg_relax
from dgfem.visualization import plot_sparsity_pattern, plot_amplification_factor
from utils.helpers import compute_m, compute_Lp_norm
from utils.logger import Logger
from utils.timer import Timer

class Relaxation:
    def __init__(self) -> None:
        self.logger = Logger(__name__).logger
        self.timer = Timer(self.logger)

    @classmethod
    def calculate_amplification(self, grid, export=False):
        theta_x, theta_y = np.linspace(-np.pi,np.pi, 101), np.linspace(-np.pi,np.pi, 101)
        Vinv = np.linalg.inv(grid.V_DOF_grid.get('u').get('u')) if grid.discretization == 'dg' else None
        
        A1 = np.zeros((len(theta_x), len(theta_y)))
        A2, A3, A4 = np.copy(A1), np.copy(A1), np.copy(A1)
        for j, theta_j in enumerate(theta_y):
            for i, theta_i in enumerate(theta_x):
                u0_modal = getattr(self, 'compute_Fourier_components_FVM' if grid.discretization=='fvm' else 'compute_Fourier_components')(grid, theta_i, theta_j, Vinv=Vinv)
                grid.BSR = grid.BSR.astype(np.complex128)
                u_modal = self.Gauss_Seidel_pyamg(grid=grid, RHS=np.zeros_like(u0_modal, dtype=np.complex128), u=u0_modal, max_iterations=1).reshape(grid.N, grid.N_DOF_sol.get('u'))#, order='F')
                      
                if grid.discretization != 'fvm':
                    u_nodal = np.einsum('ij,kj->ki', grid.V_DOF_grid.get('u').get('u'), u_modal)
                    
                    # this assumes that Ni,Nj can be divided by 2
                    m1 = compute_m(grid.Ni/2-1,grid.Nj/2-1,grid.Ni)     # lower left
                    m2 = compute_m(grid.Ni/2,grid.Nj/2-1,grid.Ni)       # lower right
                    m3 = compute_m(grid.Ni/2-1,grid.Nj/2,grid.Ni)       # upper left
                    m4 = compute_m(grid.Ni/2,grid.Nj/2,grid.Ni)         # upper right

                    A1[i,j] = abs(u_nodal)[m1,-1]
                    A2[i,j] = abs(u_nodal)[m2,-1-grid.N_sol.get('u')]
                    A3[i,j] = abs(u_nodal)[m3,grid.N_sol.get('u')]
                    A4[i,j] = abs(u_nodal)[m4,0]
                else:
                    u_nodal = u_modal.reshape(grid.Ni,grid.Nj, order='F')
                    A1[i,j] = abs(u_nodal)[int(grid.Ni/2-1), int(grid.Nj/2-1)]
                    A2[i,j] = abs(u_nodal)[int(grid.Ni/2), int(grid.Nj/2-1)]
                    A3[i,j] = abs(u_nodal)[int(grid.Ni/2-1), int(grid.Nj/2)]
                    A4[i,j] = abs(u_nodal)[int(grid.Ni/2), int(grid.Nj/2)]
                # A_mean[i,j] = (A1[i,j] + A2[i,j] + A3[i,j] + A4[i,j])/4

        print(f'{np.min(A1)=}')
        print(f'{np.max(A1)=}')
        plot_amplification_factor(A1, theta_x, theta_y, grid, export=export, suffix='0')
        # exit()
        print(f'{np.min(A2)=}')
        print(f'{np.max(A2)=}')
        plot_amplification_factor(A2, theta_x, theta_y, grid, export=export, suffix='1')
        print(f'{np.min(A3)=}')
        print(f'{np.max(A3)=}')
        plot_amplification_factor(A3, theta_x, theta_y, grid, export=export, suffix='2')
        print(f'{np.min(A4)=}')
        print(f'{np.max(A4)=}')
        plot_amplification_factor(A4, theta_x, theta_y, grid, export=export, suffix='3')
        exit()
        return u

    @classmethod
    def compute_Fourier_components(self, grid, theta_x, theta_y, Vinv):
        x0, y0 = grid.elements[0,0].x[0,0], grid.elements[0,0].y[0,0]
        xL, yL = grid.elements[-1,-1].x[-1,-1], grid.elements[-1,-1].y[-1,-1]
        Lx = xL - x0
        Ly = yL - y0
        
        Ni_tot = grid.Ni*(grid.N_grid-1)
        Nj_tot = grid.Nj*(grid.N_grid-1)
        
        k = np.array([[(grid.elements[i,j].x_rs.get('u') - x0)*Ni_tot/Lx for j in range(grid.Nj)] for i in range(grid.Ni)])
        l = np.array([[(grid.elements[i,j].y_rs.get('u') - y0)*Nj_tot/Ly for j in range(grid.Nj)] for i in range(grid.Ni)])
        
        f_nodal = np.exp(1j*(theta_x*k + theta_y*l))
        np.testing.assert_allclose(abs(f_nodal), np.full_like(f_nodal, 1.0))

        f_nodal = f_nodal.reshape((grid.Ni, grid.Nj, -1), order='F')
        f_nodal = f_nodal.reshape((grid.N, -1), order='F')
        f_modal = np.einsum('ij,kj->ki', Vinv, f_nodal)
        return np.ravel(f_modal)
    
    @classmethod
    def compute_Fourier_components_FVM(self, grid, theta_x, theta_y, **kwargs):
        k, l = np.meshgrid(np.arange(grid.Ni),np.arange(grid.Nj), indexing='ij')

        f_nodal = np.exp(1j*(theta_x*k + theta_y*l))
        np.testing.assert_allclose(abs(f_nodal), np.full_like(f_nodal, 1.0))
        
        f_nodal = np.ravel(f_nodal, order='F')
        f_modal = f_nodal
        return f_modal
    
    @classmethod
    def jacobi(self, grid, RHS, u=None, direction=None, omega=1, max_iterations=1e3):
        """Quite a different answer from Jacobi_pyamg"""
        if not isinstance(u, np.ndarray): u = np.zeros_like(RHS)
        u = u.copy()
        if not isinstance(grid.BSR_D, sp.bsr_array): grid.BSR_E, grid.BSR_D, grid.BSR_F = self.split_block_EDF(grid.BSR)
        for _ in range(max_iterations):
            u = splin.spsolve(grid.BSR_D.tocsc(), omega*((grid.BSR_E + grid.BSR_F) @ u + RHS) + grid.BSR_D.tocsc() @ u * (1-omega))
        return u
    
    @classmethod
    def jacobi_pyamg(self, grid, RHS, u=None, direction=None, omega=1., max_iterations=1e3):
        if not isinstance(u, np.ndarray): u = np.zeros_like(RHS)
        temp = u.copy()
        for _ in range(max_iterations):
            amgc.bsr_jacobi(grid.BSR.indptr, grid.BSR.indices, grid.BSR.data, u, RHS, temp, 0, len(grid.BSR.indptr)-1, 1, grid.BSR.blocksize[0], np.array([omega]))
            temp = u
        return u
 
    @classmethod
    def block_jacobi(self, grid, RHS, u=None, direction=None, omega=1, max_iterations=1e3):
        if not isinstance(u, np.ndarray): u = np.zeros_like(RHS)
        u = u.copy()
        if not isinstance(grid.BSR_D, sp.bsr_array): grid.BSR_E, grid.BSR_D, grid.BSR_F = self.split_block_EDF(grid.BSR)

        u_new = np.zeros_like(u)
        block_size = grid.BSR_D.blocksize[0]
        for _ in range(max_iterations):
            for i in range(len(grid.BSR_D.indptr)-1):
                k_E = range(grid.BSR_E.indptr[i], grid.BSR_E.indptr[i+1])
                k_D = range(grid.BSR_D.indptr[i], grid.BSR_D.indptr[i+1])
                k_F = range(grid.BSR_F.indptr[i], grid.BSR_F.indptr[i+1])

                block_cols_E = grid.BSR_E.indices[k_E]
                block_cols_D = grid.BSR_D.indices[k_D]
                block_cols_F = grid.BSR_F.indices[k_F]

                data_E = grid.BSR_E.data[k_E].transpose(1,0,2).reshape(block_size,-1)
                data_D = grid.BSR_D.data[k_D].transpose(1,0,2).reshape(block_size,-1)
                data_F = grid.BSR_F.data[k_F].transpose(1,0,2).reshape(block_size,-1)

                j_E = list(chain.from_iterable(map(lambda x: list(range(x*block_size, (x+1)*block_size)), block_cols_E)))
                j_D = list(chain.from_iterable(map(lambda x: list(range(x*block_size, (x+1)*block_size)), block_cols_D)))
                j_F = list(chain.from_iterable(map(lambda x: list(range(x*block_size, (x+1)*block_size)), block_cols_F)))

                u_new[j_D] = omega*np.linalg.solve(data_D, data_E @ u[j_E] + data_F @ u[j_F] + RHS[j_D]) + (1-omega)*u[j_D]
            u = u_new
        return u

    @classmethod
    def gauss_seidel(self, grid, RHS, u=None, direction='forward', omega=1, max_iterations=1e3):
        if not isinstance(u, np.ndarray): u = np.zeros_like(RHS)
        u = u.copy()
        if not isinstance(grid.BSR_D, sp.bsr_array): grid.BSR_E, grid.BSR_D, grid.BSR_F = self.split_block_EDF(grid.BSR)
        for _ in range(max_iterations):
            u = splin.spsolve((grid.BSR_D-grid.BSR_E).tocsc(), grid.BSR_F @ u + RHS)
        return u
    
    @classmethod
    def gauss_seidel_pyamg(self, grid, RHS, u=None, direction='symmetric', omega=1, max_iterations=1e3):
        if not isinstance(u, np.ndarray): u = np.zeros_like(RHS)
        u = u.copy()
        
        amg_relax.gauss_seidel(A=grid.BSR, x=u, b=RHS, iterations=max_iterations, sweep=direction)
        return u

    @classmethod
    def block_gauss_seidel(self, grid, RHS, u=None, direction='forward', omega=1, max_iterations=1e3):
        if not isinstance(u, np.ndarray): u = np.zeros_like(RHS)
        u = u.copy()
        if not isinstance(grid.BSR_D, sp.bsr_array): grid.BSR_E, grid.BSR_D, grid.BSR_F = self.split_block_EDF(grid.BSR)

        block_size = grid.BSR_D.blocksize[0]
        for _ in range(max_iterations):
            for i in range(len(grid.BSR_D.indptr)-1):
                k_E = range(grid.BSR_E.indptr[i], grid.BSR_E.indptr[i+1])
                k_D = range(grid.BSR_D.indptr[i], grid.BSR_D.indptr[i+1])
                k_F = range(grid.BSR_F.indptr[i], grid.BSR_F.indptr[i+1])

                block_cols_E = grid.BSR_E.indices[k_E]
                block_cols_D = grid.BSR_D.indices[k_D]
                block_cols_F = grid.BSR_F.indices[k_F]

                data_E = grid.BSR_E.data[k_E].transpose(1,0,2).reshape(block_size,-1)
                data_D = grid.BSR_D.data[k_D].transpose(1,0,2).reshape(block_size,-1)
                data_F = grid.BSR_F.data[k_F].transpose(1,0,2).reshape(block_size,-1)

                j_E = list(chain.from_iterable(map(lambda x: list(range(x*block_size, (x+1)*block_size)), block_cols_E)))
                j_D = list(chain.from_iterable(map(lambda x: list(range(x*block_size, (x+1)*block_size)), block_cols_D)))
                j_F = list(chain.from_iterable(map(lambda x: list(range(x*block_size, (x+1)*block_size)), block_cols_F)))

                u[j_D] = omega*np.linalg.solve(data_D, data_E @ u[j_E] + data_F @ u[j_F] + RHS[j_D]) + (1-omega)*u[j_D]
        return u
    
    @classmethod
    def block_gauss_seidel_pyamg(self, grid, RHS, u=None, direction='symmetric', omega=1, max_iterations=1e3):
        if not isinstance(u, np.ndarray): u = np.zeros_like(RHS)
        u = u.copy()

        residual_0 = compute_Lp_norm(RHS - grid.BSR @ u, 2)
        residuals = []

        n = 0
        while n < max_iterations:
            amg_relax.block_gauss_seidel(A=grid.BSR, x=u, b=RHS, iterations=1, sweep=direction, blocksize=grid.BSR.blocksize[0])
            residual = compute_Lp_norm(RHS - grid.BSR @ u, 2)/residual_0
            residuals.append(residual)
            # print(f"The residual (normalised) after {n} sweeps is {residual:.6e}")
            if residual<1e-6:
                print(f"Residual reduced by 6 orders in {n+1} sweeps")
                break
            elif residual>1e10:
                print(f"diverging, residual={residual:.6e}")
                exit()
            n += 1
        return u
    
    @classmethod
    def distributive_gauss_seidel(self, grid, RHS, u=None, inner_smoother='block_gauss_seidel_pyamg', splitting='classical_exact', omega=1, max_iterations=1e3, settings=None):
        if settings.problem.type != 'Stokes': raise ValueError("Distributive Gauss-Seidel is only possible for the Stokes equations")
        if settings.solution.ordering != 'global': raise ValueError("The solution ordering must be global in order to use distributive Gauss-Seidel")
        if not isinstance(u, np.ndarray): u = np.zeros_like(RHS)
        BSR_0 = grid.BSR
        BSR_0_D = grid.BSR_D
        residual_0 = compute_Lp_norm(RHS - grid.BSR @ u, 2)

        residuals_path = os.path.join(os.getcwd(), 'postprocessing', 'pickles', 'relaxation')
        residuals_file = f'residuals_{settings.problem.type}_{grid.Ni}X{grid.Nj}_nPoly{grid.P_grid}_Pu{grid.P_sol.get("u")}_Pp{grid.P_sol.get("p")}_{splitting}'
        residuals_file += '_circle' if settings.grid.circular else '_rectangle'
        residuals_file += '.pkl'
        pickle_path = os.path.join(residuals_path, residuals_file)
        residuals = []

        n = 0

        sweeps = 1
        if splitting == 'lsq':
            grid.BSR_block_DG = grid.BSR_block_D @ grid.BSR_block_G
            while n < max_iterations:
                ### building transformed RHS
                idx_u = grid.N*grid.N_DOF_sol.get('u')*2
                u_k = u[:idx_u]
                p_k = u[idx_u:]
                f_mom = RHS[:idx_u]
                f_cont = RHS[idx_u:]
                delta_u, delta_u_star = np.zeros_like(u_k), np.zeros_like(u_k)
                delta_p, delta_p_star = np.zeros_like(p_k), np.zeros_like(p_k)
                RHS_mom = f_mom - grid.BSR_block_A @ u_k - grid.BSR_block_G @ p_k

                ### solving transformed systems
                grid.BSR = grid.BSR_block_A
                delta_u_star = self.block_gauss_seidel_pyamg(grid, RHS_mom, u=delta_u_star, direction='symmetric', max_iterations=sweeps)
                
                grid.BSR = grid.BSR_block_DG
                RHS_cont = f_cont - grid.BSR_block_D @ (u_k + delta_u_star) #- grid.BSR_block_0 @ p_k
                delta_p_star = self.block_gauss_seidel_pyamg(grid, RHS_cont, u=delta_p_star, direction='symmetric', max_iterations=sweeps)

                ### transforming back to original variables
                delta_u = delta_u_star + grid.BSR_block_G @ delta_p_star

                RHS_DG = -grid.BSR_block_D @ grid.BSR_block_A @ grid.BSR_block_G @ delta_p_star
                grid.BSR = grid.BSR_block_DG
                delta_p = self.block_gauss_seidel_pyamg(grid, RHS_DG, u=delta_p, direction='symmetric', max_iterations=sweeps)

                u[:idx_u] += delta_u
                u[idx_u:] += delta_p

                residual = compute_Lp_norm(RHS - BSR_0 @ u, 2)/residual_0
                residuals.append(residual)
                # print(f"The residual (normalised) after {n} sweeps is {residual:.6e}")
                if residual<1e-6:
                    print(f"Residual reduced by 6 orders in {n} sweeps")
                    break
                elif residual>1e10:
                    print(f"diverging, residual={residual:.6e}")
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(residuals, f)
                    exit()
                n += 1
            with open(pickle_path, 'wb') as f:
                pickle.dump(residuals, f)
        elif splitting == 'classical':
            _, grid.BSR_block_A_D, _ = self.split_block_EDF(grid.BSR_block_A)
            grid.BSR_block_Ainv = splin.inv(grid.BSR_block_A_D.tocsc())     # smoother only works when Ainv is NOT approximated by inverse of block diagonal of A, otherwise it won't converge
            # grid.BSR_block_Ainv = splin.inv(grid.BSR_block_A.tocsc())
            
            grid.BSR_block_Schur = -grid.BSR_block_D @ grid.BSR_block_Ainv @ grid.BSR_block_G #+ grid.BSR_block_0
            while n < max_iterations:
                ### building transformed RHS
                idx_u = grid.N*grid.N_DOF_sol.get('u')*2
                u_k = u[:idx_u]
                p_k = u[idx_u:]
                f_mom = RHS[:idx_u]
                f_cont = RHS[idx_u:]
                delta_u, delta_u_star = np.zeros_like(u_k), np.zeros_like(u_k)
                delta_p, delta_p_star = np.zeros_like(p_k), np.zeros_like(p_k)
                RHS_mom = f_mom - grid.BSR_block_A @ u_k - grid.BSR_block_G @ p_k

                ### solving transformed systems
                grid.BSR = grid.BSR_block_A_D
                delta_u_star = self.block_gauss_seidel_pyamg(grid, RHS_mom, u=delta_u_star, direction='symmetric', max_iterations=1)
                
                
                grid.BSR = grid.BSR_block_Schur
                RHS_cont = f_cont - grid.BSR_block_D @ (u_k + delta_u_star) #- grid.BSR_block_0 @ p_k
                delta_p_star = self.block_gauss_seidel_pyamg(grid, RHS_cont, u=delta_p_star, direction='symmetric', max_iterations=1)

                ### transforming back to original variables
                RHS_A = grid.BSR_block_A @ delta_u_star - grid.BSR_block_G @ delta_p_star
                grid.BSR = grid.BSR_block_A
                delta_u = self.block_gauss_seidel_pyamg(grid, RHS_A, u=delta_u, direction='symmetric', max_iterations=1)
                delta_p = delta_p_star

                u[:idx_u] += delta_u
                u[idx_u:] += delta_p

                residual = compute_Lp_norm(RHS - BSR_0 @ u, 2)/residual_0
                residuals.append(residual)
                # print(f"The residual (normalised) after {n} sweeps is {residual:.6e}")
                if residual<1e-6:
                    print(f"Residual reduced by 6 orders in {n} sweeps")
                    break
                elif residual>1e10:
                    print(f"diverging, residual={residual:.6e}")
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(residuals, f)
                    exit()
                n += 1
            with open(pickle_path, 'wb') as f:
                pickle.dump(residuals, f)
        if splitting == 'lsq_exact': # broke this wile debugging, this is not the lsq_exact anymore but classical_exact
            pass
            # # if not isinstance(grid.BSR_block_A_D, sp.bsr_array): _, grid.BSR_block_A_D, _ = self.split_block_EDF(grid.BSR_block_A)
            # # if not isinstance(grid.BSR_block_Ainv, sp.bsr_array): grid.BSR_block_Ainv = splin.inv(grid.BSR_block_A.tocsc())
            # # if not isinstance(grid.BSR_block_DG, sp.bsr_array): 
            # #     grid.BSR_block_DG = grid.BSR_block_D @ grid.BSR_block_G
            # #     _, grid.BSR_block_DG_D, _ = self.split_block_EDF(grid.BSR_block_DG)
            
            # ### building transformed RHS
            # idx_u = grid.N*grid.N_DOF_sol.get('u')*2
            # u_k = u[:idx_u]
            # p_k = u[idx_u:]
            # f_mom = RHS[:idx_u]
            # f_cont = RHS[idx_u:]
            # delta_u, delta_u_star = np.zeros_like(u_k), np.zeros_like(u_k)
            # delta_p, delta_p_star = np.zeros_like(p_k), np.zeros_like(p_k)
            # RHS_mom = f_mom - grid.BSR_block_A @ u_k - grid.BSR_block_G @ p_k

            # ### checking PAG
            # DG = grid.BSR_block_D @ grid.BSR_block_G
            # eigs_DG, _ = np.linalg.eig(DG.toarray())
            # print(f"min(eigs_DG)={min(eigs_DG)}, max(eigs_DG)={max(eigs_DG)}")
            # DGinv = splin.inv(DG.tocsc())
            # # P = np.eye(grid.BSR_block_G.shape[0], grid.BSR_block_D.shape[1]) - grid.BSR_block_G @ DGinv @ grid.BSR_block_D
            # # PAG = P @ grid.BSR_block_A @ grid.BSR_block_G
            
            # # print(f'minimum of commutator is {np.min(PAG)}')
            # # print(f'maximum of commutator is {np.max(PAG)}')
            # # # grid.BSR = sp.bsr_array(P)
            # # # plot_sparsity_pattern(grid)
            # # # exit()

            # ### solving transformed systems
            # # delta_u_star = splin.inv(grid.BSR_block_A.tocsc()) @ RHS_mom
            
            # # test = delta_u_star
            # grid.BSR = grid.BSR_block_A
            # grid.BSR_D = grid.BSR_block_A_D
            # self.check_iteration_matrix(grid.BSR)
            # eigs_block_A, _ = np.linalg.eig(grid.BSR_block_A.toarray())
            # print(f"min(eigs(A))={min(eigs_block_A)}, max(eigs(A))={max(eigs_block_A)}")
            # delta_u_star = getattr(self, inner_smoother)(grid, RHS_mom, u=delta_u_star, max_iterations=max_iterations)
            # # np.testing.assert_allclose(test, delta_u_star)
            
            # RHS_cont = f_cont - grid.BSR_block_D @ (u_k + delta_u_star) #- grid.BSR_block_0 @ p_k
            # DAG = grid.BSR_block_D @ grid.BSR_block_A @ grid.BSR_block_G
            # LHS = DG #- grid.BSR_block_0 @ DGinv @ DAG
            # eigs_LHS, _ = np.linalg.eig(LHS.toarray())
            # print(f"min(eigs_LHS)={min(eigs_LHS)}, max(eigs_LHS)={max(eigs_LHS)}")
            # # delta_p_star = splin.inv(LHS.tocsc()) @ RHS_cont

            # grid.BSR = LHS
            # _, grid.BSR_D, _ = self.split_block_EDF(LHS)
            # self.check_iteration_matrix(grid.BSR)
            # delta_p_star = getattr(self, inner_smoother)(grid, RHS_cont, u=delta_p_star, max_iterations=max_iterations)

            # ### transforming back to original variables
            # delta_u = delta_u_star + grid.BSR_block_G @ delta_p_star

            # # RHS = -grid.BSR_block_D @ grid.BSR_block_A @ grid.BSR_block_G @ delta_p_star
            # # delta_p = DGinv @ RHS
            # delta_p = -DGinv @ DAG @ delta_p_star
            # # np.testing.assert_allclose(delta_p, delta_p_2)
            # # exit()

            # u[:idx_u] += delta_u
            # u[idx_u:] += delta_p
        elif splitting == 'classical_exact':
            if not isinstance(grid.BSR_block_A_D, sp.bsr_array): _, grid.BSR_block_A_D, _ = self.split_block_EDF(grid.BSR_block_A)
            if not isinstance(grid.BSR_block_Ainv, sp.bsr_array): grid.BSR_block_Ainv = splin.inv(grid.BSR_block_A.tocsc())     # smoother only works when Ainv is NOT approximated by inverse of block diagonal of A, otherwise it won't converge
            if not isinstance(grid.BSR_block_Schur, sp.bsr_array): 
                grid.BSR_block_Schur = -grid.BSR_block_D @ grid.BSR_block_Ainv @ grid.BSR_block_G + grid.BSR_block_0
                _, grid.BSR_block_Schur_D, _ = self.split_block_EDF(grid.BSR_block_Schur)
            
            ### building transformed RHS
            idx_u = grid.N*grid.N_DOF_sol.get('u')*2
            u_k = u[:idx_u]
            p_k = u[idx_u:]
            f_mom = RHS[:idx_u]
            f_cont = RHS[idx_u:]
            delta_u, delta_u_star = np.zeros_like(u_k), np.zeros_like(u_k)
            delta_p, delta_p_star = np.zeros_like(p_k), np.zeros_like(p_k)
            RHS_mom = f_mom - grid.BSR_block_A @ u_k - grid.BSR_block_G @ p_k

            ### solving transformed systems
            grid.BSR = grid.BSR_block_A
            grid.BSR_D = grid.BSR_block_A_D
            delta_u_star = getattr(self, inner_smoother)(grid, RHS_mom, u=delta_u_star, max_iterations=max_iterations)
            # delta_u_star = splin.inv(grid.BSR_block_A) @ RHS_mom
            
            grid.BSR = grid.BSR_block_Schur
            grid.BSR_D = grid.BSR_block_Schur_D 
            RHS_cont = f_cont - grid.BSR_block_D @ (u_k + delta_u_star) - grid.BSR_block_0 @ p_k
            delta_p_star = getattr(self, inner_smoother)(grid, RHS_cont, u=delta_p_star, max_iterations=max_iterations)
            # delta_p_star = splin.inv(grid.BSR_block_Schur) @ RHS_cont

            ### transforming back to original variables
            RHS = grid.BSR_block_A @ delta_u_star - grid.BSR_block_G @ delta_p_star
            grid.BSR = grid.BSR_block_A
            grid.BSR_D = grid.BSR_block_A_D
            delta_u = getattr(self, inner_smoother)(grid, RHS, u=delta_u, max_iterations=max_iterations)
            # delta_u = splin.inv(grid.BSR_block_Schur) @ RHS_cont
            delta_p = delta_p_star

            u[:idx_u] += delta_u
            u[idx_u:] += delta_p
        grid.BSR = BSR_0
        grid.BSR_D = BSR_0_D
        return u
    
    @classmethod
    def split_block_EDF(self, BSR):
        """This is NOT a LDU factorization"""
        data_E, indices_E, indptr_E = [], [], [0]
        data_D, indices_D, indptr_D = [], [], [0]
        data_F, indices_F, indptr_F = [], [], [0]
        for i in range(len(BSR.indptr)-1):
            k1 = BSR.indptr[i]
            k2 = BSR.indptr[i+1]
            idx = list(range(k1, k2))

            indices_row = BSR.indices[k1:k2]
            indices_row_E = list(filter(lambda j: j < i, indices_row))
            indices_row_D = list(filter(lambda j: j == i, indices_row))
            indices_row_F = list(filter(lambda j: j > i, indices_row))

            idx_E = [idx[i] for i, j in enumerate(indices_row) if j in indices_row_E]
            idx_D = [idx[i] for i, j in enumerate(indices_row) if j in indices_row_D]
            idx_F = [idx[i] for i, j in enumerate(indices_row) if j in indices_row_F]

            data_row_E = -BSR.data[idx_E]
            data_row_D = BSR.data[idx_D]
            data_row_F = -BSR.data[idx_F]

            data_E.extend(data_row_E)
            indices_E.extend(indices_row_E)
            indptr_E.append(indptr_E[-1]+len(indices_row_E))
            
            data_D.extend(data_row_D)

            indices_D.extend(indices_row_D)
            indptr_D.append(indptr_D[-1]+len(indices_row_D))
            
            data_F.extend(data_row_F)
            indices_F.extend(indices_row_F)
            indptr_F.append(indptr_F[-1]+len(indices_row_F))

        E = sp.bsr_array((data_E, indices_E, indptr_E), shape=BSR.shape, blocksize=BSR.blocksize) if indices_E else sp.bsr_array(BSR.shape)
        D = sp.bsr_array((data_D, indices_D, indptr_D), shape=BSR.shape, blocksize=BSR.blocksize) if indices_D else sp.bsr_array(BSR.shape)
        F = sp.bsr_array((data_F, indices_F, indptr_F), shape=BSR.shape, blocksize=BSR.blocksize) if indices_F else sp.bsr_array(BSR.shape)

        # np.testing.assert_allclose(BSR.todense(), (BSR_E+BSR_D+BSR_F).todense())
        # plot_sparsity_pattern([BSR, BSR_E, BSR_D, BSR_F, BSR_D+BSR_F+BSR_E])

        # if not len(E.data): E = sp.bsr_array(E.shape, dtype=E.dtype)
        # if not len(E.data): F = sp.bsr_array(E.shape, dtype=E.dtype)
        
        # if E.blocksize != D.blocksize or E.blocksize != F.blocksize or D.blocksize != F.blocksize: raise ValueError("E, D and F blocks are not of the same size")
        # if (E.indptr != D.indptr).all() or (E.indptr != F.indptr).all() or (D.indptr != F.indptr).all(): raise ValueError("E, D and F BSR are not of the same size")
        return E, D, F

    @classmethod
    def check_iteration_matrix(self, BSR, which='forward_Gauss_Seidel', omega=1.):
        E, D, F = self.split_block_EDF(BSR)
        if which=='forward_Gauss_Seidel':
            B = splin.inv(D.tocsc()-E.tocsc()) @ F
        if which=='backward_Gauss_Seidel':
            B = splin.inv(D.tocsc()-F.tocsc()) @ E
        elif which=='SOR':
            I = sp.identity(D.shape[0])
            Dinv = splin.inv(D.tocsc())
            B = splin.inv((I - omega*Dinv @ E).tocsc())*((1-omega)*I + omega*Dinv @ F)
        elif which=='Jacobi':
            B = splin.inv(D.tocsc()) @ (E+F)

        max_eig = abs(splin.eigs(B.tocsc(), k=1, which='LM', return_eigenvectors=False)[0])
        print(f"The max eigenvalue of {which} iteration matrix B is {max_eig:.3e}")