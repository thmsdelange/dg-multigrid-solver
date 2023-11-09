import numpy as np
import scipy.sparse as sp
import os
import pickle

from dgfem.relaxation import Relaxation
from dgfem.visualization import plot_sparsity_pattern
from utils.helpers import compute_m, is_diagonally_dominant, obj_to_dict
from utils.logger import Logger

class DiscreteSystem:
    def __init__(self, settings) -> None:
        self.problem = self.select_problem(settings)

    def select_problem(self, settings):
        if settings.problem.type.lower() == 'poisson':
            return Poisson(settings)
        elif settings.problem.type.lower() == 'stokes':
            return Stokes(settings)
        

class Poisson:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.logger = Logger(__name__, settings).logger

    def assemble(self, grid):
        if grid.discretization == 'dg':
            discrete_system_path = os.path.join(os.getcwd(), 'cache', 'discrete_system')
            if not grid.coarsening_factor: 
                discrete_system_file = f'discrete_system_{self.settings.problem.type}_{grid.Ni}X{grid.Nj}_nPoly{grid.P_grid}_pSol{grid.P_sol.get("u")}'
            else:
                discrete_system_file = f'discrete_system_{self.settings.problem.type}_{grid.Ni_fine}X{grid.Nj_fine}_nPoly{grid.P_grid}_pSol{grid.P_sol.get("u")}_coarsened_{grid.coarsening_factor}'
            if self.settings.grid.circular: discrete_system_file += '_circle'
            discrete_system_file += '.pkl'
            pickle_path = os.path.join(discrete_system_path, discrete_system_file)
            if not os.path.exists(pickle_path) or not self.settings.caching.enabled:
                self.assemble_BSR_Poisson(grid)
                self.assemble_RHS_Poisson(grid)
                if self.settings.caching.enabled:
                    with open(pickle_path, 'wb') as f:
                        pickle.dump({'BSR': grid.BSR, 'RHS': grid.RHS, 'settings': self.settings}, f)
            else:
                with open(pickle_path, 'rb') as f:
                    pickle_dict = pickle.load(f)
                assert obj_to_dict(self.settings.grid) == obj_to_dict(pickle_dict.get('settings').grid)
                assert obj_to_dict(self.settings.solution) == obj_to_dict(pickle_dict.get('settings').solution)
                assert obj_to_dict(self.settings.problem) == obj_to_dict(pickle_dict.get('settings').problem)
                grid.BSR = pickle_dict.get('BSR')
                grid.RHS = pickle_dict.get('RHS')
        elif grid.discretization == 'fvm':
            self.assemble_BSR_and_RHS_Poisson_FVM(grid)

    def assemble_BSR_Poisson(self, grid):
        if self.settings.problem.type != 'Poisson': raise ValueError("The governing equation(s) field in the paramfile is not set to Poisson but the assemble_global_BSR is called for the Poisson problem")
        if not grid.coarsening_factor:
            self.logger.debug(f"Assembling the coefficient matrix for grid with P_grid={grid.P_grid}, P_sol={grid.P_sol} and sigma={grid.sigma}")
        else:
            self.logger.debug(f"Assembling the coefficient matrix for coarsened grid (by a factor {grid.coarsening_factor}) with P_grid={grid.P_grid}, P_sol={grid.P_sol} and sigma={grid.sigma}")

        data = []
        indices = []
        indptr = [0]
        for j in range(grid.Nj):
            for i in range(grid.Ni):
                m = compute_m(i,j, grid.Ni)
                val_m = grid.elements[i,j].compute_momentum_laplace_volume_integral(self.settings.problem.type)

                face_imin = grid.faces_i[i,j]
                face_imax = grid.faces_i[i+1,j]
                face_jmin = grid.faces_j[i,j]
                face_jmax = grid.faces_j[i,j+1]

                _, _, face_imin_res_RL, face_imin_res_RR = face_imin.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                face_imax_res_LL, face_imax_res_LR, _, _ = face_imax.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                _, _, face_jmin_res_RL, face_jmin_res_RR = face_jmin.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                face_jmax_res_LL, face_jmax_res_LR, _, _ = face_jmax.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                val_m += face_imin_res_RR
                val_m += face_imax_res_LL
                val_m += face_jmin_res_RR
                val_m += face_jmax_res_LL
                
                if i==0: 
                    if grid.O_grid or grid.fully_periodic_boundaries: # periodic with i==Ni-1
                        m_iL = compute_m(grid.Ni-1,j, grid.Ni)
                        val_iL = face_imin_res_RL
                    else: # Dirichlet boundary
                        m_iL = None
                        val_iL = None
                else:
                    m_iL = compute_m(i-1,j, grid.Ni)
                    val_iL = face_imin_res_RL
                
                if i==grid.Ni-1: 
                    if grid.O_grid or grid.fully_periodic_boundaries: # periodic with i==0
                        m_iR = compute_m(0,j, grid.Ni)
                        val_iR = face_imax_res_LR
                    else: # Dirichlet boundary
                        m_iR = None
                        val_iR = None
                else:
                    m_iR = compute_m(i+1,j, grid.Ni)
                    val_iR = face_imax_res_LR

                if j==0: 
                    if grid.fully_periodic_boundaries: # periodic with j==Nj-1
                        m_jL = compute_m(i,grid.Nj-1, grid.Ni)
                        val_jL = face_jmin_res_RL
                    else: # Dirichlet boundary
                        m_jL = None
                        val_jL = None
                else:
                    m_jL = compute_m(i,j-1, grid.Ni)
                    val_jL = face_jmin_res_RL
                
                if j==grid.Nj-1: 
                    if grid.fully_periodic_boundaries: # periodic with j==0
                        m_jR = compute_m(i,0, grid.Ni)
                        val_jR = face_jmax_res_LR
                    else: # Dirichlet boundary
                        m_jR = None
                        val_jR = None
                else:
                    m_jR = compute_m(i,j+1, grid.Ni)
                    val_jR = face_jmax_res_LR

                mass_matrix = grid.elements[i,j].compute_mass_matrix()
                grid.elements[i,j].inv_mass_matrix = np.linalg.inv(mass_matrix)

                if self.settings.problem.check_eigenvalues:
                    eigenvals, _ = np.linalg.eig(mass_matrix)
                    if min(eigenvals)<0 or max(eigenvals)<0:
                        self.logger.critical("The mass matrix is not SPD (both positive and negative eigenvalues)")

                ### building data structures for the Block Compressed Sparse Row (Block CSR or BSR) matrix
                indices_row = [elem for elem in [m, m_iL, m_iR, m_jL, m_jR] if elem is not None]
                indices.extend(sorted(indices_row))
                sorted_indices_row = sorted(range(len(indices_row)), key=lambda k: indices_row[k])
                if self.settings.problem.multiply_inverse_mass_matrix:
                    data_row = [grid.elements[i,j].inv_mass_matrix @ val for val in [val_m, val_iL, val_iR, val_jL, val_jR] if val is not None]
                else:
                    data_row = [val for val in [val_m, val_iL, val_iR, val_jL, val_jR] if val is not None]
                data.extend([data_row[i] for i in sorted_indices_row])
                indptr.append(indptr[-1]+len(indices_row))
        grid.BSR = sp.bsr_array((data, indices, indptr), shape=(grid.Ni*grid.Nj*grid.N_DOF_sol_tot, grid.Ni*grid.Nj*grid.N_DOF_sol_tot))
        
        if self.settings.problem.check_eigenvalues:
            import scipy.sparse.linalg as splin
            min_eig = splin.eigs(grid.BSR.tocsc(), k=1, which='SR', return_eigenvectors=False)[0]
            max_eig = splin.eigs(grid.BSR.tocsc(), k=1, which='LR', return_eigenvectors=False)[0]
            self.logger.debug(f"The eigenvalues of the coefficient matrix are {min_eig:.2e} (min) and {max_eig:.2e} (max)")
        
        if self.settings.problem.check_characteristics:
            A = grid.BSR.toarray()
            try:
                ### check if the coefficient matrix is symmetric positive definite (SPD)
                np.testing.assert_allclose(A, A.T, atol=1e-13)
            except:
                self.logger.warning("The Poisson system is NOT SPD, not symmetric")
            
            try:
                ### a good test for positive definiteness is to try to compute its Cholesky factorization. It succeeds if matrix is positive definite.
                np.linalg.cholesky(A)
                self.logger.debug("The Poisson system is SPD")
            except:
                self.logger.warning("The Poisson system is NOT SPD, not positive definite")

            dd = is_diagonally_dominant(A)
            if dd:
                self.logger.debug("The Poisson system is diagonally dominant")
            else:
                self.logger.warning("The Poisson system is NOT diagonally dominant")
            exit()

        if self.settings.problem.check_orthonormality:
            for j in range(grid.Nj):
                for i in range(grid.Ni):
                    mass_matrix = grid.elements[i,j].inv_mass_matrix
                    mass_matrix[mass_matrix<1e-10] = 0.
                    print(f"Mass matrix of element {i},{j}:\n {mass_matrix}\n")

        if self.settings.problem.check_iteration_matrix:
            Relaxation.check_iteration_matrix(grid.BSR)
            exit()

        if self.settings.visualization.plot_sparsity_pattern: plot_sparsity_pattern(grid)

    def assemble_BSR_and_RHS_Poisson_FVM(self, grid, bc_order=2):
        """
         +------jmax------+----------------+
         |                |                |
         |                |                |
        imin    (i,j)    imax   (i+1,j)    |    Delta_s = s_i+1,j - s_i,j
         |                |                |
         |                |                |
         +------jmin------+----------------+
                          |
                        s_imax
        """
        # if self.P_sol.get('u') != 0: raise ValueError("The FV method can only be used when P_sol=0")
        if self.settings.problem.type != 'Poisson': raise ValueError("The FV method can currently only be done for the Poisson problem")

        RHS = np.zeros((grid.N))
        self.u_exact_nodal = np.zeros((grid.N))
        data = []
        indices = []
        indptr = [0]
        for j in range(grid.Nj):
            for i in range(grid.Ni):
                

                m = compute_m(i,j, grid.Ni)
                element_ij = grid.elements[i,j]
                x_ij, y_ij = element_ij.metric_xy_rs(element_ij.x, element_ij.y, [0], [0])
                f_ij = self.settings.problem.exact_solution_function(x_ij, y_ij, self.settings.problem.type, which='source_momentum')[0]
                RHS[m] -= f_ij.get('u')*element_ij.A
                self.u_exact_nodal[m] = self.settings.problem.exact_solution_function(x_ij, y_ij, self.settings.problem.type, which='solution').get('u').get('u')

                if i==0: 
                    if grid.O_grid: # periodic with i==Ni-1
                        m_iL = compute_m(grid.Ni-1,j, grid.Ni)
                        element_iL = grid.elements[grid.Ni-1,j]
                    elif not grid.O_grid: # Dirichlet boundary
                        m_iL = None
                        element_iL = None
                else:
                    m_iL = compute_m(i-1,j, grid.Ni)
                    element_iL = grid.elements[i-1,j]
                
                if i==grid.Ni-1: 
                    if grid.O_grid: # periodic with i==0
                        m_iR = compute_m(0,j, grid.Ni)
                        element_iR = grid.elements[0,j]
                    elif not grid.O_grid: # Dirichlet boundary
                        m_iR = None
                        element_iR = None
                else:
                    m_iR = compute_m(i+1,j, grid.Ni)
                    element_iR = grid.elements[i+1,j]

                if j==0: # Dirichlet boundary
                    m_jL = None
                    element_jL = None
                else:
                    m_jL = compute_m(i,j-1, grid.Ni)
                    element_jL = grid.elements[i,j-1]
                
                if j==grid.Nj-1: # Dirichlet boundary
                    m_jR = None
                    element_jR = None
                else:
                    m_jR = compute_m(i,j+1, grid.Ni)
                    element_jR = grid.elements[i,j+1]

                x_iL, y_iL = element_iL.metric_xy_rs(element_iL.x, element_iL.y, [0], [0]) if element_iL else element_ij.metric_xy_rs(element_ij.x, element_ij.y, [-1], [0])
                x_iR, y_iR = element_iR.metric_xy_rs(element_iR.x, element_iR.y, [0], [0]) if element_iR else element_ij.metric_xy_rs(element_ij.x, element_ij.y, [1], [0])
                x_jL, y_jL = element_jL.metric_xy_rs(element_jL.x, element_jL.y, [0], [0]) if element_jL else element_ij.metric_xy_rs(element_ij.x, element_ij.y, [0], [-1])
                x_jR, y_jR = element_jR.metric_xy_rs(element_jR.x, element_jR.y, [0], [0]) if element_jR else element_ij.metric_xy_rs(element_ij.x, element_ij.y, [0], [1])

                Delta_s_imin = np.linalg.norm(np.array([x_ij.get('u') - x_iL.get('u'), y_ij.get('u') - y_iL.get('u')]))     # Delta_s corresponding to face imin
                Delta_s_imax = np.linalg.norm(np.array([x_iR.get('u') - x_ij.get('u'), y_iR.get('u') - y_ij.get('u')]))     # Delta_s corresponding to face imax
                Delta_s_jmin = np.linalg.norm(np.array([x_ij.get('u') - x_jL.get('u'), y_ij.get('u') - y_jL.get('u')]))     # Delta_s corresponding to face jmin
                Delta_s_jmax = np.linalg.norm(np.array([x_jR.get('u') - x_ij.get('u'), y_jR.get('u') - y_ij.get('u')]))     # Delta_s corresponding to face jmax

                x_imin_jmin, y_imin_jmin = element_ij.metric_xy_rs(element_ij.x, element_ij.y, [-1], [-1])
                x_imin_jmax, y_imin_jmax = element_ij.metric_xy_rs(element_ij.x, element_ij.y, [-1], [1])
                x_imax_jmin, y_imax_jmin = element_ij.metric_xy_rs(element_ij.x, element_ij.y, [1], [-1])
                x_imax_jmax, y_imax_jmax = element_ij.metric_xy_rs(element_ij.x, element_ij.y, [1], [1])

                s_imin = np.linalg.norm(np.array([x_imin_jmax.get('u') - x_imin_jmin.get('u'), y_imin_jmax.get('u') - y_imin_jmin.get('u')]))
                s_imax = np.linalg.norm(np.array([x_imax_jmax.get('u') - x_imax_jmin.get('u'), y_imax_jmax.get('u') - y_imax_jmin.get('u')]))
                s_jmin = np.linalg.norm(np.array([x_imax_jmin.get('u') - x_imin_jmin.get('u'), y_imax_jmin.get('u') - y_imin_jmin.get('u')]))
                s_jmax = np.linalg.norm(np.array([x_imax_jmax.get('u') - x_imin_jmax.get('u'), y_imax_jmax.get('u') - y_imin_jmax.get('u')]))
                
                a_ij = 0.
                if element_iL:
                    a_iL = s_imin/Delta_s_imin 
                    a_ij -= a_iL
                else:
                    a_iL = None
                    a_iLB = bc_order*s_imin/(2*Delta_s_imin)
                    u_iLB = self.settings.problem.exact_solution_function(x_iL, y_iL, self.settings.problem.type, which='solution').get('u').get('u')[0]
                    RHS[m] -= a_iLB*u_iLB
                    a_ij -= a_iLB
                if element_iR:
                    a_iR = s_imax/Delta_s_imax 
                    a_ij -= a_iR
                else:
                    a_iR = None
                    a_iRB = bc_order*s_imax/(2*Delta_s_imax)
                    u_iRB = self.settings.problem.exact_solution_function(x_iR, y_iR, self.settings.problem.type, which='solution').get('u').get('u')[0]
                    RHS[m] -= a_iRB*u_iRB
                    a_ij -= a_iRB
                if element_jL:
                    a_jL = s_jmin/Delta_s_jmin 
                    a_ij -= a_jL
                else:
                    a_jL = None
                    a_jLB = bc_order*s_jmin/(2*Delta_s_jmin)
                    u_jLB = self.settings.problem.exact_solution_function(x_jL, y_jL, self.settings.problem.type, which='solution').get('u').get('u')[0]
                    RHS[m] -= a_jLB*u_jLB
                    a_ij -= a_jLB
                if element_jR:
                    a_jR = s_jmax/Delta_s_jmax 
                    a_ij -= a_jR
                else:
                    a_jR = None
                    a_jRB = bc_order*s_jmax/(2*Delta_s_jmax)
                    u_jRB = self.settings.problem.exact_solution_function(x_jR, y_jR, self.settings.problem.type, which='solution').get('u').get('u')[0]
                    RHS[m] -= a_jRB*u_jRB
                    a_ij -= a_jRB

                indices_row = [elem for elem in [m, m_iL, m_iR, m_jL, m_jR] if elem is not None]
                indices.extend(sorted(indices_row))
                sorted_indices_row = sorted(range(len(indices_row)), key=lambda k: indices_row[k])
                data_row = [val for val in [a_ij, a_iL, a_iR, a_jL, a_jR] if val is not None]
                data.extend([data_row[i] for i in sorted_indices_row])
                indptr.append(indptr[-1]+len(indices_row))
        grid.BSR = sp.csr_array((data, indices, indptr), shape=(grid.N,grid.N))
        grid.RHS = RHS


        if self.settings.problem.check_characteristics:
            A = grid.BSR.toarray()
            print(A)
            try:
                ### check if the coefficient matrix is symmetric positive definite (SPD)
                np.testing.assert_allclose(A, A.T, atol=1e-13)
            except:
                self.logger.warning("The Poisson system is NOT SPD, not symmetric")
            
            try:
                ### a good test for positive definiteness is to try to compute its Cholesky factorization. It succeeds if matrix is positive definite.
                np.linalg.cholesky(A)
                self.logger.debug("The Poisson system is SPD")
            except:
                eigenvalues, _ = np.linalg.eig(grid.BSR.toarray())
                min_eig, max_eig = min(eigenvalues), max(eigenvalues)
                if min_eig<0 and max_eig>0:
                    self.logger.debug(f"The eigenvalues of the coefficient matrix are {min_eig:.5g} (min) and {max_eig:.5g} (max)")
                elif min_eig<0 and max_eig<0:
                    self.logger.debug(f"The eigenvalues of the coefficient matrix are {min_eig:.5g} (min) and {max_eig:.5g} (max)")
                    self.logger.debug("The Poisson system is SND, symmetric negative definite")
                else:
                    self.logger.debug(f"The eigenvalues of the coefficient matrix are {min_eig:.5g} (min) and {max_eig:.5g} (max)")
                self.logger.warning("The Poisson system is NOT SPD, not positive definite")

            dd = is_diagonally_dominant(A)
            if dd:
                self.logger.debug("The Poisson system is diagonally dominant")
            else:
                self.logger.warning("The Poisson system is NOT diagonally dominant")
            exit()

    def assemble_RHS_Poisson(self, grid):
        if not grid.coarsening_factor:
            self.logger.debug(f"Assembling the RHS vector for grid with P_grid={grid.P_grid}, P_sol={grid.P_sol} and sigma={grid.sigma}")
        else:
            self.logger.debug(f"Assembling the RHS vector for coarsened grid (by a factor {grid.coarsening_factor}) with P_grid={grid.P_grid}, P_sol={grid.P_sol} and sigma={grid.sigma}")

        RHS = np.zeros((grid.Ni*grid.Nj*grid.N_DOF_sol.get('u')))
        for j in range(grid.Nj):
            for i in range(grid.Ni):
                m = compute_m(i,j,grid.Ni)
                start = m*grid.N_DOF_sol.get('u')
                end = (m+1)*grid.N_DOF_sol.get('u')

                element = grid.elements[i,j]
                face_imin = grid.faces_i[i,j]
                face_imax = grid.faces_i[i+1,j]
                face_jmin = grid.faces_j[i,j]
                face_jmax = grid.faces_j[i,j+1]
                xy_int = element.xy_int

                f_int = self.settings.problem.exact_solution_function(*xy_int.get('xy_int'), self.settings.problem.type, 'source_momentum')
                RHS[start:end] += element.compute_source_momentum_volume_integral(self.settings.problem.type, f_int)
                
                if not grid.fully_periodic_boundaries:
                    if not grid.O_grid:
                        if i==0: # Dirichlet boundary
                            g_int = self.settings.problem.exact_solution_function(*xy_int['xy_int_imin'], self.settings.problem.type, 'solution')
                            RHS[start:end] += face_imin.compute_momentum_laplace_SIP_penalty_term(self.settings.problem.type, RHS=True, g_int=g_int)
                            RHS[start:end] += face_imin.compute_momentum_laplace_SIP_symmetrizing_term(self.settings.problem.type, RHS=True, g_int=g_int)
                        if i==grid.Ni-1: # Dirichlet boundary
                            g_int = self.settings.problem.exact_solution_function(*xy_int['xy_int_imax'], self.settings.problem.type, 'solution')
                            RHS[start:end] += face_imax.compute_momentum_laplace_SIP_penalty_term(self.settings.problem.type, RHS=True, g_int=g_int)
                            RHS[start:end] += face_imax.compute_momentum_laplace_SIP_symmetrizing_term(self.settings.problem.type, RHS=True, g_int=g_int)

                    if j==0: # Dirichlet boundary
                        g_int = self.settings.problem.exact_solution_function(*xy_int['xy_int_jmin'], self.settings.problem.type, 'solution')
                        RHS[start:end] += face_jmin.compute_momentum_laplace_SIP_penalty_term(self.settings.problem.type, RHS=True, g_int=g_int)
                        RHS[start:end] += face_jmin.compute_momentum_laplace_SIP_symmetrizing_term(self.settings.problem.type, RHS=True, g_int=g_int)
                    if j==grid.Nj-1: # Dirichlet boundary
                        g_int = self.settings.problem.exact_solution_function(*xy_int['xy_int_jmax'], self.settings.problem.type, 'solution')
                        RHS[start:end] += face_jmax.compute_momentum_laplace_SIP_penalty_term(self.settings.problem.type, RHS=True, g_int=g_int)
                        RHS[start:end] += face_jmax.compute_momentum_laplace_SIP_symmetrizing_term(self.settings.problem.type, RHS=True, g_int=g_int)

                if self.settings.problem.multiply_inverse_mass_matrix:
                    if not isinstance(element.inv_mass_matrix, np.ndarray):
                        mass_matrix = element.compute_mass_matrix()
                        element.inv_mass_matrix = np.linalg.inv(mass_matrix)
                    RHS[start:end] = element.inv_mass_matrix @ RHS[start:end]
        grid.RHS = RHS

class Stokes:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.logger = Logger(__name__, settings).logger
        

    def assemble(self, grid):
        ### assemble BSR and RHS
        getattr(self, f'assemble_BSR_Stokes_{self.settings.solution.ordering.lower()}_order')(grid)
        self.assemble_RHS_Stokes(grid)

    def assemble_BSR_Stokes_global_order(self, grid):
        if not grid.coarsening_factor:
            self.logger.debug(f"Assembling the coefficient matrix for grid with P_grid={grid.P_grid}, P_sol={grid.P_sol} and sigma={grid.sigma}")
        else:
            self.logger.debug(f"Assembling the coefficient matrix for coarsened grid (by a factor {grid.coarsening_factor}) with P_grid={grid.P_grid}, P_sol={grid.P_sol} and sigma={grid.sigma}")

        data_Au_xmom = []
        data_Au_ymom = []
        data_Av_xmom = []
        data_Av_ymom = []
        data_Du = []
        data_Dv = []
        data_Gx = []
        data_Gy = []
        indices = []
        indptr = [0]
        for j in range(grid.Nj):
            for i in range(grid.Ni):
                ### initializing 
                m = compute_m(i,j, grid.Ni)
                face_imin = grid.faces_i[i,j]
                face_imax = grid.faces_i[i+1,j]
                face_jmin = grid.faces_j[i,j]
                face_jmax = grid.faces_j[i,j+1]

                # f_int_continuity = self.settings.problem.exact_solution_function(*grid.elements[i,j].xy_int.get('xy_int'), self.settings.problem.type, which='source_continuity')
                # f_cont = grid.elements[i,j].compute_source_continuity_volume_integral(f_int_continuity)
                # print(f_cont)
                # print(f_int_continuity)
                
                val_Au_xmom_m = np.zeros((grid.N_DOF_sol.get('u'),grid.N_DOF_sol.get('u')))
                val_Au_xmom_iL, val_Au_xmom_iR, val_Au_xmom_jL, val_Au_xmom_jR = np.copy(val_Au_xmom_m), np.copy(val_Au_xmom_m), np.copy(val_Au_xmom_m), np.copy(val_Au_xmom_m)
                val_Au_ymom_m = np.zeros((grid.N_DOF_sol.get('u'),grid.N_DOF_sol.get('u')))
                val_Au_ymom_iL, val_Au_ymom_iR, val_Au_ymom_jL, val_Au_ymom_jR = np.copy(val_Au_ymom_m), np.copy(val_Au_ymom_m), np.copy(val_Au_ymom_m), np.copy(val_Au_ymom_m)
                val_Av_xmom_m = np.zeros((grid.N_DOF_sol.get('u'),grid.N_DOF_sol.get('u')))
                val_Av_xmom_iL, val_Av_xmom_iR, val_Av_xmom_jL, val_Av_xmom_jR = np.copy(val_Av_xmom_m), np.copy(val_Av_xmom_m), np.copy(val_Av_xmom_m), np.copy(val_Av_xmom_m)
                val_Av_ymom_m = np.zeros((grid.N_DOF_sol.get('u'),grid.N_DOF_sol.get('u')))
                val_Av_ymom_iL, val_Av_ymom_iR, val_Av_ymom_jL, val_Av_ymom_jR = np.copy(val_Av_ymom_m), np.copy(val_Av_ymom_m), np.copy(val_Av_ymom_m), np.copy(val_Av_ymom_m)
                
                val_Du_m = np.zeros((grid.N_DOF_sol.get('p'),grid.N_DOF_sol.get('u')))
                val_Du_iL, val_Du_iR, val_Du_jL, val_Du_jR = np.copy(val_Du_m), np.copy(val_Du_m), np.copy(val_Du_m), np.copy(val_Du_m)
                val_Dv_m = np.zeros((grid.N_DOF_sol.get('p'),grid.N_DOF_sol.get('u')))
                val_Dv_iL, val_Dv_iR, val_Dv_jL, val_Dv_jR = np.copy(val_Dv_m), np.copy(val_Dv_m), np.copy(val_Dv_m), np.copy(val_Dv_m)
                
                val_Gx_m = np.zeros((grid.N_DOF_sol.get('u'),grid.N_DOF_sol.get('p')))
                val_Gx_iL, val_Gx_iR, val_Gx_jL, val_Gx_jR = np.copy(val_Gx_m), np.copy(val_Gx_m), np.copy(val_Gx_m), np.copy(val_Gx_m)
                val_Gy_m = np.zeros((grid.N_DOF_sol.get('u'),grid.N_DOF_sol.get('p')))
                val_Gy_iL, val_Gy_iR, val_Gy_jL, val_Gy_jR = np.copy(val_Gy_m), np.copy(val_Gy_m), np.copy(val_Gy_m), np.copy(val_Gy_m)
                
                ### u,v contributions from x,y momentum (A)
                contrib_momentum_laplace_volume_integral = grid.elements[i,j].compute_momentum_laplace_volume_integral(self.settings.problem.type)
                val_Au_xmom_m += contrib_momentum_laplace_volume_integral[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')] 
                val_Av_ymom_m += contrib_momentum_laplace_volume_integral[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                contrib_momentum_velocity_penalty_volume_integral = grid.elements[i,j].compute_momentum_velocity_penalty_volume_integral()
                val_Au_xmom_m += contrib_momentum_velocity_penalty_volume_integral[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                val_Au_ymom_m += contrib_momentum_velocity_penalty_volume_integral[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')] 
                val_Av_xmom_m += contrib_momentum_velocity_penalty_volume_integral[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                val_Av_ymom_m += contrib_momentum_velocity_penalty_volume_integral[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]
                # print(contrib_momentum_velocity_penalty_volume_integral)

                _, _, contrib_face_imin_SIP_res_RL, contrib_face_imin_SIP_res_RR = face_imin.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                contrib_face_imax_SIP_res_LL, contrib_face_imax_SIP_res_LR, _, _ = face_imax.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                _, _, contrib_face_jmin_SIP_res_RL, contrib_face_jmin_SIP_res_RR = face_jmin.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                contrib_face_jmax_SIP_res_LL, contrib_face_jmax_SIP_res_LR, _, _ = face_jmax.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                val_Au_xmom_m += contrib_face_imin_SIP_res_RR[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')] 
                val_Au_xmom_m += contrib_face_imax_SIP_res_LL[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')] 
                val_Au_xmom_m += contrib_face_jmin_SIP_res_RR[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')] 
                val_Au_xmom_m += contrib_face_jmax_SIP_res_LL[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')] 

                val_Au_ymom_m += contrib_face_imin_SIP_res_RR[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')] 
                val_Au_ymom_m += contrib_face_imax_SIP_res_LL[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')] 
                val_Au_ymom_m += contrib_face_jmin_SIP_res_RR[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')] 
                val_Au_ymom_m += contrib_face_jmax_SIP_res_LL[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')] 

                val_Av_xmom_m += contrib_face_imin_SIP_res_RR[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                val_Av_xmom_m += contrib_face_imax_SIP_res_LL[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                val_Av_xmom_m += contrib_face_jmin_SIP_res_RR[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                val_Av_xmom_m += contrib_face_jmax_SIP_res_LL[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]

                val_Av_ymom_m += contrib_face_imin_SIP_res_RR[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]
                val_Av_ymom_m += contrib_face_imax_SIP_res_LL[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]
                val_Av_ymom_m += contrib_face_jmin_SIP_res_RR[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]
                val_Av_ymom_m += contrib_face_jmax_SIP_res_LL[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                _, _, contrib_face_imin_velocity_penalty_res_RL, contrib_face_imin_velocity_penalty_res_RR = face_imin.compute_momentum_velocity_penalty_surface_integral()
                contrib_face_imax_velocity_penalty_res_LL, contrib_face_imax_velocity_penalty_res_LR, _, _ = face_imax.compute_momentum_velocity_penalty_surface_integral()
                _, _, contrib_face_jmin_velocity_penalty_res_RL, contrib_face_jmin_velocity_penalty_res_RR = face_jmin.compute_momentum_velocity_penalty_surface_integral()
                contrib_face_jmax_velocity_penalty_res_LL, contrib_face_jmax_velocity_penalty_res_LR, _, _ = face_jmax.compute_momentum_velocity_penalty_surface_integral()

                val_Au_xmom_m += contrib_face_imin_velocity_penalty_res_RR[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')] 
                val_Au_xmom_m += contrib_face_imax_velocity_penalty_res_LL[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')] 
                val_Au_xmom_m += contrib_face_jmin_velocity_penalty_res_RR[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')] 
                val_Au_xmom_m += contrib_face_jmax_velocity_penalty_res_LL[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')] 

                val_Au_ymom_m += contrib_face_imin_velocity_penalty_res_RR[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')] 
                val_Au_ymom_m += contrib_face_imax_velocity_penalty_res_LL[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')] 
                val_Au_ymom_m += contrib_face_jmin_velocity_penalty_res_RR[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')] 
                val_Au_ymom_m += contrib_face_jmax_velocity_penalty_res_LL[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')] 

                val_Av_xmom_m += contrib_face_imin_velocity_penalty_res_RR[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                val_Av_xmom_m += contrib_face_imax_velocity_penalty_res_LL[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                val_Av_xmom_m += contrib_face_jmin_velocity_penalty_res_RR[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                val_Av_xmom_m += contrib_face_jmax_velocity_penalty_res_LL[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]

                val_Av_ymom_m += contrib_face_imin_velocity_penalty_res_RR[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]
                val_Av_ymom_m += contrib_face_imax_velocity_penalty_res_LL[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]
                val_Av_ymom_m += contrib_face_jmin_velocity_penalty_res_RR[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]
                val_Av_ymom_m += contrib_face_jmax_velocity_penalty_res_LL[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                ### u,v contributions from continuity (D)
                contrib_continuity_volume_integral = grid.elements[i,j].compute_continuity_volume_integral()
                val_Du_m += contrib_continuity_volume_integral[:,:grid.N_DOF_sol.get('u')]
                val_Dv_m += contrib_continuity_volume_integral[:,-grid.N_DOF_sol.get('u'):]

                _, _, contrib_face_imin_continuity_res_RL, contrib_face_imin_continuity_res_RR = face_imin.compute_continuity_surface_integral()
                contrib_face_imax_continuity_res_LL, contrib_face_imax_continuity_res_LR, _, _ = face_imax.compute_continuity_surface_integral()
                _, _, contrib_face_jmin_continuity_res_RL, contrib_face_jmin_continuity_res_RR = face_jmin.compute_continuity_surface_integral()
                contrib_face_jmax_continuity_res_LL, contrib_face_jmax_continuity_res_LR, _, _ = face_jmax.compute_continuity_surface_integral()
                val_Du_m += contrib_face_imin_continuity_res_RR[:,:grid.N_DOF_sol.get('u')]
                val_Du_m += contrib_face_imax_continuity_res_LL[:,:grid.N_DOF_sol.get('u')]
                val_Du_m += contrib_face_jmin_continuity_res_RR[:,:grid.N_DOF_sol.get('u')]
                val_Du_m += contrib_face_jmax_continuity_res_LL[:,:grid.N_DOF_sol.get('u')]

                val_Dv_m += contrib_face_imin_continuity_res_RR[:,-grid.N_DOF_sol.get('u'):]
                val_Dv_m += contrib_face_imax_continuity_res_LL[:,-grid.N_DOF_sol.get('u'):]
                val_Dv_m += contrib_face_jmin_continuity_res_RR[:,-grid.N_DOF_sol.get('u'):]
                val_Dv_m += contrib_face_jmax_continuity_res_LL[:,-grid.N_DOF_sol.get('u'):]

                ### p contribution from x,y momentum (G)
                contrib_momentum_pressure_volume_integral = grid.elements[i,j].compute_momentum_pressure_volume_integral()
                val_Gx_m += contrib_momentum_pressure_volume_integral[:grid.N_DOF_sol.get('u'),:]
                val_Gy_m += contrib_momentum_pressure_volume_integral[-grid.N_DOF_sol.get('u'):,:]
                
                _, _, contrib_face_imin_pressure_res_RL, contrib_face_imin_pressure_res_RR = face_imin.compute_momentum_pressure_surface_integral()
                contrib_face_imax_pressure_res_LL, contrib_face_imax_pressure_res_LR, _, _ = face_imax.compute_momentum_pressure_surface_integral()
                _, _, contrib_face_jmin_pressure_res_RL, contrib_face_jmin_pressure_res_RR = face_jmin.compute_momentum_pressure_surface_integral()
                contrib_face_jmax_pressure_res_LL, contrib_face_jmax_pressure_res_LR, _, _ = face_jmax.compute_momentum_pressure_surface_integral()
                val_Gx_m += contrib_face_imin_pressure_res_RR[:grid.N_DOF_sol.get('u'),:]
                val_Gx_m += contrib_face_imax_pressure_res_LL[:grid.N_DOF_sol.get('u'),:]
                val_Gx_m += contrib_face_jmin_pressure_res_RR[:grid.N_DOF_sol.get('u'),:]
                val_Gx_m += contrib_face_jmax_pressure_res_LL[:grid.N_DOF_sol.get('u'),:]

                val_Gy_m += contrib_face_imin_pressure_res_RR[-grid.N_DOF_sol.get('u'):,:]
                val_Gy_m += contrib_face_imax_pressure_res_LL[-grid.N_DOF_sol.get('u'):,:]
                val_Gy_m += contrib_face_jmin_pressure_res_RR[-grid.N_DOF_sol.get('u'):,:]
                val_Gy_m += contrib_face_jmax_pressure_res_LL[-grid.N_DOF_sol.get('u'):,:]
           
                if i==0: 
                    if grid.O_grid: # periodic with i==Ni-1
                        m_iL = compute_m(grid.Ni-1,j, grid.Ni)

                        val_Au_xmom_iL += contrib_face_imin_SIP_res_RL[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                        val_Au_ymom_iL += contrib_face_imin_SIP_res_RL[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                        val_Av_xmom_iL += contrib_face_imin_SIP_res_RL[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                        val_Av_ymom_iL += contrib_face_imin_SIP_res_RL[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                        val_Au_xmom_iL += contrib_face_imin_velocity_penalty_res_RL[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                        val_Au_ymom_iL += contrib_face_imin_velocity_penalty_res_RL[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                        val_Av_xmom_iL += contrib_face_imin_velocity_penalty_res_RL[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                        val_Av_ymom_iL += contrib_face_imin_velocity_penalty_res_RL[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                        val_Du_iL += contrib_face_imin_continuity_res_RL[:,:grid.N_DOF_sol.get('u')]
                        val_Dv_iL += contrib_face_imin_continuity_res_RL[:,-grid.N_DOF_sol.get('u'):]
                        
                        val_Gx_iL += contrib_face_imin_pressure_res_RL[:grid.N_DOF_sol.get('u'),:]
                        val_Gy_iL += contrib_face_imin_pressure_res_RL[-grid.N_DOF_sol.get('u'):,:]
                    elif not grid.O_grid: # Dirichlet boundary
                        m_iL = None
                        val_Au_xmom_iL, val_Au_ymom_iL, val_Av_xmom_iL, val_Av_ymom_iL = None, None, None, None
                        val_Du_iL, val_Dv_iL = None, None
                        val_Gx_iL, val_Gy_iL = None, None
                else:
                    m_iL = compute_m(i-1,j, grid.Ni)

                    val_Au_xmom_iL += contrib_face_imin_SIP_res_RL[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                    val_Au_ymom_iL += contrib_face_imin_SIP_res_RL[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                    val_Av_xmom_iL += contrib_face_imin_SIP_res_RL[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                    val_Av_ymom_iL += contrib_face_imin_SIP_res_RL[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                    val_Au_xmom_iL += contrib_face_imin_velocity_penalty_res_RL[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                    val_Au_ymom_iL += contrib_face_imin_velocity_penalty_res_RL[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                    val_Av_xmom_iL += contrib_face_imin_velocity_penalty_res_RL[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                    val_Av_ymom_iL += contrib_face_imin_velocity_penalty_res_RL[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                    val_Du_iL += contrib_face_imin_continuity_res_RL[:,:grid.N_DOF_sol.get('u')]
                    val_Dv_iL += contrib_face_imin_continuity_res_RL[:,-grid.N_DOF_sol.get('u'):]

                    val_Gx_iL += contrib_face_imin_pressure_res_RL[:grid.N_DOF_sol.get('u'),:]
                    val_Gy_iL += contrib_face_imin_pressure_res_RL[-grid.N_DOF_sol.get('u'):,:]

                if i==grid.Ni-1: 
                    if grid.O_grid: # periodic with i==0
                        m_iR = compute_m(0,j, grid.Ni)

                        val_Au_xmom_iR += contrib_face_imax_SIP_res_LR[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                        val_Au_ymom_iR += contrib_face_imax_SIP_res_LR[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                        val_Av_xmom_iR += contrib_face_imax_SIP_res_LR[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                        val_Av_ymom_iR += contrib_face_imax_SIP_res_LR[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                        val_Au_xmom_iR += contrib_face_imax_velocity_penalty_res_LR[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                        val_Au_ymom_iR += contrib_face_imax_velocity_penalty_res_LR[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                        val_Av_xmom_iR += contrib_face_imax_velocity_penalty_res_LR[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                        val_Av_ymom_iR += contrib_face_imax_velocity_penalty_res_LR[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                        val_Du_iR += contrib_face_imax_continuity_res_LR[:,:grid.N_DOF_sol.get('u')]
                        val_Dv_iR += contrib_face_imax_continuity_res_LR[:,-grid.N_DOF_sol.get('u'):]

                        val_Gx_iR += contrib_face_imax_pressure_res_LR[:grid.N_DOF_sol.get('u'),:]
                        val_Gy_iR += contrib_face_imax_pressure_res_LR[-grid.N_DOF_sol.get('u'):,:]
                    elif not grid.O_grid: # Dirichlet boundary
                        m_iR = None
                        val_Au_xmom_iR, val_Au_ymom_iR, val_Av_xmom_iR, val_Av_ymom_iR = None, None, None, None
                        val_Du_iR, val_Dv_iR = None, None
                        val_Gx_iR, val_Gy_iR = None, None
                else:
                    m_iR = compute_m(i+1,j, grid.Ni)

                    val_Au_xmom_iR += contrib_face_imax_SIP_res_LR[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                    val_Au_ymom_iR += contrib_face_imax_SIP_res_LR[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                    val_Av_xmom_iR += contrib_face_imax_SIP_res_LR[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                    val_Av_ymom_iR += contrib_face_imax_SIP_res_LR[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                    val_Au_xmom_iR += contrib_face_imax_velocity_penalty_res_LR[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                    val_Au_ymom_iR += contrib_face_imax_velocity_penalty_res_LR[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                    val_Av_xmom_iR += contrib_face_imax_velocity_penalty_res_LR[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                    val_Av_ymom_iR += contrib_face_imax_velocity_penalty_res_LR[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                    val_Du_iR += contrib_face_imax_continuity_res_LR[:,:grid.N_DOF_sol.get('u')]
                    val_Dv_iR += contrib_face_imax_continuity_res_LR[:,-grid.N_DOF_sol.get('u'):]

                    val_Gx_iR += contrib_face_imax_pressure_res_LR[:grid.N_DOF_sol.get('u'),:]
                    val_Gy_iR += contrib_face_imax_pressure_res_LR[-grid.N_DOF_sol.get('u'):,:]

                if j==0: # Dirichlet boundary
                    m_jL = None
                    val_Au_xmom_jL, val_Au_ymom_jL, val_Av_xmom_jL, val_Av_ymom_jL = None, None, None, None
                    val_Du_jL, val_Dv_jL = None, None
                    val_Gx_jL, val_Gy_jL = None, None
                else:
                    m_jL = compute_m(i,j-1, grid.Ni)
                    
                    val_Au_xmom_jL += contrib_face_jmin_SIP_res_RL[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                    val_Au_ymom_jL += contrib_face_jmin_SIP_res_RL[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                    val_Av_xmom_jL += contrib_face_jmin_SIP_res_RL[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                    val_Av_ymom_jL += contrib_face_jmin_SIP_res_RL[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                    val_Au_xmom_jL += contrib_face_jmin_velocity_penalty_res_RL[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                    val_Au_ymom_jL += contrib_face_jmin_velocity_penalty_res_RL[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                    val_Av_xmom_jL += contrib_face_jmin_velocity_penalty_res_RL[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                    val_Av_ymom_jL += contrib_face_jmin_velocity_penalty_res_RL[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                    val_Du_jL += contrib_face_jmin_continuity_res_RL[:,:grid.N_DOF_sol.get('u')]
                    val_Dv_jL += contrib_face_jmin_continuity_res_RL[:,-grid.N_DOF_sol.get('u'):]
                    
                    val_Gx_jL += contrib_face_jmin_pressure_res_RL[:grid.N_DOF_sol.get('u'),:]
                    val_Gy_jL += contrib_face_jmin_pressure_res_RL[-grid.N_DOF_sol.get('u'):,:]
                
                if j==grid.Nj-1: # Dirichlet boundary
                    m_jR = None
                    val_Au_xmom_jR, val_Au_ymom_jR, val_Av_xmom_jR, val_Av_ymom_jR = None, None, None, None
                    val_Du_jR, val_Dv_jR = None, None
                    val_Gx_jR, val_Gy_jR = None, None
                else:
                    m_jR = compute_m(i,j+1, grid.Ni)

                    val_Au_xmom_jR += contrib_face_jmax_SIP_res_LR[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                    val_Au_ymom_jR += contrib_face_jmax_SIP_res_LR[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                    val_Av_xmom_jR += contrib_face_jmax_SIP_res_LR[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                    val_Av_ymom_jR += contrib_face_jmax_SIP_res_LR[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                    val_Au_xmom_jR += contrib_face_jmax_velocity_penalty_res_LR[:grid.N_DOF_sol.get('u'),:grid.N_DOF_sol.get('u')]
                    val_Au_ymom_jR += contrib_face_jmax_velocity_penalty_res_LR[-grid.N_DOF_sol.get('u'):,:grid.N_DOF_sol.get('u')]
                    val_Av_xmom_jR += contrib_face_jmax_velocity_penalty_res_LR[:grid.N_DOF_sol.get('u'),-grid.N_DOF_sol.get('u'):]
                    val_Av_ymom_jR += contrib_face_jmax_velocity_penalty_res_LR[-grid.N_DOF_sol.get('u'):,-grid.N_DOF_sol.get('u'):]

                    val_Du_jR += contrib_face_jmax_continuity_res_LR[:,:grid.N_DOF_sol.get('u')]
                    val_Dv_jR += contrib_face_jmax_continuity_res_LR[:,-grid.N_DOF_sol.get('u'):]

                    val_Gx_jR += contrib_face_jmax_pressure_res_LR[:grid.N_DOF_sol.get('u'),:]
                    val_Gy_jR += contrib_face_jmax_pressure_res_LR[-grid.N_DOF_sol.get('u'):,:] 

                # mass_matrix = grid.elements[i,j].compute_mass_matrix()
                # grid.elements[i,j].inv_mass_matrix = np.linalg.inv(mass_matrix)

                ### building data structures for the Block Compressed Sparse Row (Block CSR or BSR) matrix
                indices_row = [elem for elem in [m, m_iL, m_iR, m_jL, m_jR] if elem is not None]
                indices.extend(sorted(indices_row))
                sorted_indices_row = sorted(range(len(indices_row)), key=lambda k: indices_row[k])
                # data_row = [grid.elements[i,j].inv_mass_matrix @ val for val in [val_m, val_iL, val_iR, val_jL, val_jR] if val is not None]
                data_Au_xmom_row = [val for val in [val_Au_xmom_m, val_Au_xmom_iL, val_Au_xmom_iR, val_Au_xmom_jL, val_Au_xmom_jR] if val is not None]
                data_Au_xmom.extend([data_Au_xmom_row[i] for i in sorted_indices_row])
                data_Au_ymom_row = [val for val in [val_Au_ymom_m, val_Au_ymom_iL, val_Au_ymom_iR, val_Au_ymom_jL, val_Au_ymom_jR] if val is not None]
                data_Au_ymom.extend([data_Au_ymom_row[i] for i in sorted_indices_row])
                data_Av_xmom_row = [val for val in [val_Av_xmom_m, val_Av_xmom_iL, val_Av_xmom_iR, val_Av_xmom_jL, val_Av_xmom_jR] if val is not None]
                data_Av_xmom.extend([data_Av_xmom_row[i] for i in sorted_indices_row])
                data_Av_ymom_row = [val for val in [val_Av_ymom_m, val_Av_ymom_iL, val_Av_ymom_iR, val_Av_ymom_jL, val_Av_ymom_jR] if val is not None]
                data_Av_ymom.extend([data_Av_ymom_row[i] for i in sorted_indices_row])
                
                data_Du_row = [val for val in [val_Du_m, val_Du_iL, val_Du_iR, val_Du_jL, val_Du_jR] if val is not None]
                data_Du.extend([data_Du_row[i] for i in sorted_indices_row])
                data_Dv_row = [val for val in [val_Dv_m, val_Dv_iL, val_Dv_iR, val_Dv_jL, val_Dv_jR] if val is not None]
                data_Dv.extend([data_Dv_row[i] for i in sorted_indices_row])
                
                data_Gx_row = [val for val in [val_Gx_m, val_Gx_iL, val_Gx_iR, val_Gx_jL, val_Gx_jR] if val is not None]
                data_Gx.extend([data_Gx_row[i] for i in sorted_indices_row])
                data_Gy_row = [val for val in [val_Gy_m, val_Gy_iL, val_Gy_iR, val_Gy_jL, val_Gy_jR] if val is not None]
                data_Gy.extend([data_Gy_row[i] for i in sorted_indices_row])
                
                indptr.append(indptr[-1]+len(indices_row))
        
        BSR_Au_xmom = sp.bsr_array((data_Au_xmom, indices, indptr), shape=(grid.N*grid.N_DOF_sol.get('u'), grid.N*grid.N_DOF_sol.get('u')))
        BSR_Au_ymom = sp.bsr_array((data_Au_ymom, indices, indptr), shape=(grid.N*grid.N_DOF_sol.get('u'), grid.N*grid.N_DOF_sol.get('u')))
        BSR_Av_xmom = sp.bsr_array((data_Av_xmom, indices, indptr), shape=(grid.N*grid.N_DOF_sol.get('u'), grid.N*grid.N_DOF_sol.get('u')))
        BSR_Av_ymom = sp.bsr_array((data_Av_ymom, indices, indptr), shape=(grid.N*grid.N_DOF_sol.get('u'), grid.N*grid.N_DOF_sol.get('u')))
        grid.BSR_block_A = sp.bsr_array(sp.vstack([sp.hstack([BSR_Au_xmom, BSR_Av_xmom]), sp.hstack([BSR_Au_ymom, BSR_Av_ymom])], format='bsr'))
        
        BSR_Du = sp.bsr_array((data_Du, indices, indptr), shape=(grid.N*grid.N_DOF_sol.get('p'), grid.N*grid.N_DOF_sol.get('u')))
        BSR_Dv = sp.bsr_array((data_Dv, indices, indptr), shape=(grid.N*grid.N_DOF_sol.get('p'), grid.N*grid.N_DOF_sol.get('u')))
        grid.BSR_block_D = sp.bsr_array(sp.hstack([BSR_Du, BSR_Dv], format='bsr'))
        
        BSR_Gx = sp.bsr_array((data_Gx, indices, indptr), shape=(grid.N*grid.N_DOF_sol.get('u'), grid.N*grid.N_DOF_sol.get('p')))
        BSR_Gy = sp.bsr_array((data_Gy, indices, indptr), shape=(grid.N*grid.N_DOF_sol.get('u'), grid.N*grid.N_DOF_sol.get('p')))
        grid.BSR_block_G = sp.bsr_array(sp.vstack([BSR_Gx, BSR_Gy], format='bsr'))
        
        block_0 = np.zeros((grid.N*grid.N_DOF_sol.get('p'),grid.N*grid.N_DOF_sol.get('p')))
        if self.settings.solver.method == 'direct': block_0[0,0] = 1.
        grid.BSR_block_0 = sp.bsr_array(block_0)

        grid.BSR = sp.bsr_array(sp.vstack([sp.hstack([grid.BSR_block_A, grid.BSR_block_G]), sp.hstack([grid.BSR_block_D, grid.BSR_block_0])], format='bsr'))
        self.logger.debug(f"Number of nonzero entries in BSR: {grid.BSR.count_nonzero()}")
        
        # try:
        #     np.testing.assert_allclose(grid.BSR.toarray(), grid.BSR.toarray().T, atol=1e-6, rtol=0)
        # except:
        #     self.logger.warning("The global coefficient matrix is NOT symmetric")

        if self.settings.problem.check_condition_number:
            self.logger.debug(f"The condition number of the coefficient matrix is {np.linalg.cond(grid.BSR.toarray()):.5g}")
        
        if self.settings.problem.check_eigenvalues:
            eigenvalues, _ = np.linalg.eig(grid.BSR.toarray())
            min_eig, max_eig = min(eigenvalues), max(eigenvalues)
            self.logger.debug(f"The eigenvalues of the coefficient matrix are {min_eig:.5g} (min) and {max_eig:.5g} (max)")
            
            eigenvalues, _ = np.linalg.eig(grid.BSR_block_A.toarray())
            min_eig, max_eig = min(eigenvalues), max(eigenvalues)
            self.logger.debug(f"The eigenvalues of the A matrix are {min_eig:.5g} (min) and {max_eig:.5g} (max)")

        if self.settings.problem.check_characteristics:
            A = grid.BSR.toarray()
            try:
                ### check if the coefficient matrix is symmetric positive definite (SPD)
                np.testing.assert_allclose(A, A.T, atol=1e-13)
            except:
                self.logger.warning("The Stokes system is NOT SPD, not symmetric")
            
            try:
                ### a good test for positive definiteness is to try to compute its Cholesky factorization. It succeeds if matrix is positive definite.
                np.linalg.cholesky(A)
                self.logger.debug("The Stokes system is SPD")
            except:
                self.logger.warning("The Stokes system is NOT SPD, not positive definite")

            dd = is_diagonally_dominant(A)
            if dd:
                self.logger.debug("The Stokes system is diagonally dominant")
            else:
                self.logger.warning("The Stokes system is NOT diagonally dominant")
            
            A = grid.BSR_block_A.toarray()
            try:
                ### check if the coefficient matrix is symmetric positive definite (SPD)
                np.testing.assert_allclose(A, A.T, atol=1e-13)
            except:
                self.logger.warning("The Stokes A system is NOT SPD, not symmetric")
            
            eigenvalues, _ = np.linalg.eig(A)
            min_eig, max_eig = min(eigenvalues), max(eigenvalues)
            self.logger.debug(f"The eigenvalues of the A matrix are {min_eig:.5g} (min) and {max_eig:.5g} (max)")
            try:
                ### a good test for positive definiteness is to try to compute its Cholesky factorization. It succeeds if matrix is positive definite.
                np.linalg.cholesky(A)
                self.logger.debug("The Stokes A system is SPD")
            except:
                self.logger.warning("The Stokes A system is NOT SPD, not positive definite")

            dd = is_diagonally_dominant(A)
            if dd:
                self.logger.debug("The Stokes A system is diagonally dominant")
            else:
                self.logger.warning("The Stokes A system is NOT diagonally dominant")
            # exit()

        if self.settings.visualization.plot_sparsity_pattern: plot_sparsity_pattern(grid)

    def assemble_BSR_Stokes_local_order(self, grid):
        if self.settings.problem.type != 'Stokes': raise ValueError("The governing equation(s) field in the paramfile is not set to Stokes but the assemble_global_BSR is called for the Stokes problem")
        if not grid.coarsening_factor:
            self.logger.debug(f"Assembling the coefficient matrix for grid with P_grid={grid.P_grid}, P_sol={grid.P_sol} and sigma={grid.sigma}")
        else:
            self.logger.debug(f"Assembling the coefficient matrix for coarsened grid (by a factor {grid.coarsening_factor}) with P_grid={grid.P_grid}, P_sol={grid.P_sol} and sigma={grid.sigma}")

        data = []
        indices = []
        indptr = [0]
        for j in range(grid.Nj):
            for i in range(grid.Ni):
                m = compute_m(i,j, grid.Ni)
                val_m = np.zeros((grid.N_DOF_sol_tot,grid.N_DOF_sol_tot))
                val_iL, val_iR, val_jL, val_jR = np.copy(val_m), np.copy(val_m), np.copy(val_m), np.copy(val_m)

                val_m[-grid.N_DOF_sol.get('p'):,:-grid.N_DOF_sol.get('p')] += grid.elements[i,j].compute_continuity_volume_integral()
                val_m[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += grid.elements[i,j].compute_momentum_laplace_volume_integral(self.settings.problem.type)
                val_m[:-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p'):] += grid.elements[i,j].compute_momentum_pressure_volume_integral()
                val_m[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += grid.elements[i,j].compute_momentum_velocity_penalty_volume_integral()

                face_imin = grid.faces_i[i,j]
                face_imax = grid.faces_i[i+1,j]
                face_jmin = grid.faces_j[i,j]
                face_jmax = grid.faces_j[i,j+1]

                _, _, face_imin_continuity_res_RL, face_imin_continuity_res_RR = face_imin.compute_continuity_surface_integral()
                face_imax_continuity_res_LL, face_imax_continuity_res_LR, _, _ = face_imax.compute_continuity_surface_integral()
                _, _, face_jmin_continuity_res_RL, face_jmin_continuity_res_RR = face_jmin.compute_continuity_surface_integral()
                face_jmax_continuity_res_LL, face_jmax_continuity_res_LR, _, _ = face_jmax.compute_continuity_surface_integral()
                val_m[-grid.N_DOF_sol.get('p'):,:-grid.N_DOF_sol.get('p')] += face_imin_continuity_res_RR
                val_m[-grid.N_DOF_sol.get('p'):,:-grid.N_DOF_sol.get('p')] += face_imax_continuity_res_LL
                val_m[-grid.N_DOF_sol.get('p'):,:-grid.N_DOF_sol.get('p')] += face_jmin_continuity_res_RR
                val_m[-grid.N_DOF_sol.get('p'):,:-grid.N_DOF_sol.get('p')] += face_jmax_continuity_res_LL

                _, _, face_imin_SIP_res_RL, face_imin_SIP_res_RR = face_imin.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                face_imax_SIP_res_LL, face_imax_SIP_res_LR, _, _ = face_imax.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                _, _, face_jmin_SIP_res_RL, face_jmin_SIP_res_RR = face_jmin.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                face_jmax_SIP_res_LL, face_jmax_SIP_res_LR, _, _ = face_jmax.compute_momentum_laplace_SIP_terms(self.settings.problem.type)
                val_m[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imin_SIP_res_RR
                val_m[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imax_SIP_res_LL
                val_m[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_jmin_SIP_res_RR
                val_m[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_jmax_SIP_res_LL
                
                _, _, face_imin_pressure_res_RL, face_imin_pressure_res_RR = face_imin.compute_momentum_pressure_surface_integral()
                face_imax_pressure_res_LL, face_imax_pressure_res_LR, _, _ = face_imax.compute_momentum_pressure_surface_integral()
                _, _, face_jmin_pressure_res_RL, face_jmin_pressure_res_RR = face_jmin.compute_momentum_pressure_surface_integral()
                face_jmax_pressure_res_LL, face_jmax_pressure_res_LR, _, _ = face_jmax.compute_momentum_pressure_surface_integral()
                val_m[:-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p'):] += face_imin_pressure_res_RR
                val_m[:-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p'):] += face_imax_pressure_res_LL
                val_m[:-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p'):] += face_jmin_pressure_res_RR
                val_m[:-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p'):] += face_jmax_pressure_res_LL
                
                _, _, face_imin_velocity_penalty_res_RL, face_imin_velocity_penalty_res_RR = face_imin.compute_momentum_velocity_penalty_surface_integral()
                face_imax_velocity_penalty_res_LL, face_imax_velocity_penalty_res_LR, _, _ = face_imax.compute_momentum_velocity_penalty_surface_integral()
                _, _, face_jmin_velocity_penalty_res_RL, face_jmin_velocity_penalty_res_RR = face_jmin.compute_momentum_velocity_penalty_surface_integral()
                face_jmax_velocity_penalty_res_LL, face_jmax_velocity_penalty_res_LR, _, _ = face_jmax.compute_momentum_velocity_penalty_surface_integral()

                val_m[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imin_velocity_penalty_res_RR
                val_m[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imax_velocity_penalty_res_LL
                val_m[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_jmin_velocity_penalty_res_RR
                val_m[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_jmax_velocity_penalty_res_LL
                
                if i==0: 
                    if grid.O_grid: # periodic with i==Ni-1
                        m_iL = compute_m(grid.Ni-1,j, grid.Ni)
                        val_iL[-grid.N_DOF_sol.get('p'):,:-grid.N_DOF_sol.get('p')] += face_imin_continuity_res_RL
                        val_iL[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imin_SIP_res_RL
                        val_iL[:-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p'):] += face_imin_pressure_res_RL
                        val_iL[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imin_velocity_penalty_res_RL
                    elif not grid.O_grid: # Dirichlet boundary
                        m_iL = None
                        val_iL = None
                else:
                    m_iL = compute_m(i-1,j, grid.Ni)
                    val_iL[-grid.N_DOF_sol.get('p'):,:-grid.N_DOF_sol.get('p')] += face_imin_continuity_res_RL
                    val_iL[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imin_SIP_res_RL
                    val_iL[:-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p'):] += face_imin_pressure_res_RL
                    val_iL[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imin_velocity_penalty_res_RL
                
                if i==grid.Ni-1: 
                    if grid.O_grid: # periodic with i==0
                        m_iR = compute_m(0,j, grid.Ni)
                        val_iR[-grid.N_DOF_sol.get('p'):,:-grid.N_DOF_sol.get('p')] += face_imax_continuity_res_LR
                        val_iR[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imax_SIP_res_LR
                        val_iR[:-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p'):] += face_imax_pressure_res_LR
                        val_iR[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imax_velocity_penalty_res_LR
                    elif not grid.O_grid: # Dirichlet boundary
                        m_iR = None
                        val_iR = None
                else:
                    m_iR = compute_m(i+1,j, grid.Ni)
                    val_iR[-grid.N_DOF_sol.get('p'):,:-grid.N_DOF_sol.get('p')] += face_imax_continuity_res_LR
                    val_iR[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imax_SIP_res_LR
                    val_iR[:-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p'):] += face_imax_pressure_res_LR
                    val_iR[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_imax_velocity_penalty_res_LR

                if j==0: # Dirichlet boundary
                    m_jL = None
                    val_jL = None
                else:
                    m_jL = compute_m(i,j-1, grid.Ni)
                    val_jL[-grid.N_DOF_sol.get('p'):,:-grid.N_DOF_sol.get('p')] += face_jmin_continuity_res_RL
                    val_jL[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_jmin_SIP_res_RL
                    val_jL[:-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p'):] += face_jmin_pressure_res_RL
                    val_jL[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_jmin_velocity_penalty_res_RL
                
                if j==grid.Nj-1: # Dirichlet boundary
                    m_jR = None
                    val_jR = None
                else:
                    m_jR = compute_m(i,j+1, grid.Ni)
                    val_jR[-grid.N_DOF_sol.get('p'):,:-grid.N_DOF_sol.get('p')] += face_jmax_continuity_res_LR
                    val_jR[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_jmax_SIP_res_LR
                    val_jR[:-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p'):] += face_jmax_pressure_res_LR
                    val_jR[:-grid.N_DOF_sol.get('p'),:-grid.N_DOF_sol.get('p')] += face_jmax_velocity_penalty_res_LR

                mass_matrix = grid.elements[i,j].compute_mass_matrix()
                grid.elements[i,j].inv_mass_matrix = np.linalg.inv(mass_matrix)

                if self.settings.problem.check_eigenvalues:
                    eigenvals, _ = np.linalg.eig(mass_matrix)
                    if min(eigenvals)<0 or max(eigenvals)<0:
                        self.logger.critical("The mass matrix is not SPD (both positive and negative eigenvalues)")

                ### building data structures for the Block Compressed Sparse Row (Block CSR or BSR) matrix
                indices_row = [elem for elem in [m, m_iL, m_iR, m_jL, m_jR] if elem is not None]
                indices.extend(sorted(indices_row))
                sorted_indices_row = sorted(range(len(indices_row)), key=lambda k: indices_row[k])
                # data_row = [grid.elements[i,j].inv_mass_matrix @ val for val in [val_m, val_iL, val_iR, val_jL, val_jR] if val is not None]
                data_row = [val for val in [val_m, val_iL, val_iR, val_jL, val_jR] if val is not None]
                data.extend([data_row[i] for i in sorted_indices_row])
                indptr.append(indptr[-1]+len(indices_row))
        ### setting a single pressure DOF to 1. since the pressure is determined up to a constant
        if self.settings.solver.method == 'direct': data[0][-grid.N_DOF_sol.get('p'),-grid.N_DOF_sol.get('p')] = 1.
        grid.BSR = sp.bsr_array((data, indices, indptr), shape=(grid.Ni*grid.Nj*grid.N_DOF_sol_tot, grid.Ni*grid.Nj*grid.N_DOF_sol_tot))
        # np.testing.assert_allclose(grid.BSR.toarray(), grid.BSR.toarray().T, atol=1e-6, rtol=0)

        if self.settings.problem.check_condition_number:
            self.logger.debug(f"The condition number of the coefficient matrix is {np.linalg.cond(grid.BSR.toarray()):.5g}")
        if self.settings.problem.check_eigenvalues:
            eigenvalues, _ = np.linalg.eig(grid.BSR.toarray())
            min_eig, max_eig = min(eigenvalues), max(eigenvalues)
            kappa = abs(max_eig/min_eig)
            kappa_from_numpy = np.linalg.cond(grid.BSR.toarray())
            if min_eig<0 and max_eig>0:
                self.logger.critical("The system of equations is not symmetric positive definite! (both positive and negative eigenvalues)")
                self.logger.debug(f"The eigenvalues of the coefficient matrix are {min_eig:.5g} (min) and {max_eig:.5g} (max)")
                self.logger.debug(f"The condition number of the coefficient matrix is {kappa:.5g} (from eigenvals), {kappa_from_numpy:.5g} (from numpy.cond)")
            else:
                self.logger.debug(f"The eigenvalues of the coefficient matrix are {min_eig:.5g} (min) and {max_eig:.5g} (max)")
                self.logger.debug(f"The condition number of the coefficient matrix is {kappa:.5g} (from eigenvals), {kappa_from_numpy:.5g} (from numpy.cond)")

        if self.settings.visualization.plot_sparsity_pattern: plot_sparsity_pattern(grid)

    def assemble_RHS_Stokes(self, grid):
        if not grid.coarsening_factor:
            self.logger.debug(f"Assembling the RHS vector for grid with P_grid={grid.P_grid}, P_sol={grid.P_sol} and sigma={grid.sigma}")
        else:
            self.logger.debug(f"Assembling the RHS vector for coarsened grid (by a factor {grid.coarsening_factor}) with P_grid={grid.P_grid}, P_sol={grid.P_sol} and sigma={grid.sigma}")

        RHS = np.zeros((grid.Ni*grid.Nj*grid.N_DOF_sol_tot))
        for j in range(grid.Nj):
            for i in range(grid.Ni):
                m = compute_m(i,j,grid.Ni)
                start = m*grid.N_DOF_sol_tot
                end = (m+1)*grid.N_DOF_sol_tot
                include_pressure_BC = self.settings.problem.include_pressure_BC

                element = grid.elements[i,j]
                face_imin = grid.faces_i[i,j]
                face_imax = grid.faces_i[i+1,j]
                face_jmin = grid.faces_j[i,j]
                face_jmax = grid.faces_j[i,j+1]

                xy_int = element.xy_int

                f_int_continuity = self.settings.problem.exact_solution_function(*xy_int.get('xy_int'), self.settings.problem.type, 'source_continuity') #if not self.settings.solution.manufactured_solution else self.settings.problem.MMS_source_continuity_function(*xy_int.get('xy_int'), grid)
                f_int_momentum = self.settings.problem.exact_solution_function(*xy_int.get('xy_int'), self.settings.problem.type, 'source_momentum')
                RHS[end-grid.N_DOF_sol.get('p'):end] += element.compute_source_continuity_volume_integral(f_int_continuity)
                RHS[start:end-grid.N_DOF_sol.get('p')] += element.compute_source_momentum_volume_integral(self.settings.problem.type, f_int_momentum)
                
                if not grid.O_grid:
                    if i==0: # Dirichlet boundary
                        g_int = self.settings.problem.exact_solution_function(*xy_int['xy_int_imin'], self.settings.problem.type, 'solution')
                        RHS[end-grid.N_DOF_sol.get('p'):end] += face_imin.compute_continuity_surface_integral(RHS=True, g_int=g_int)
                        RHS[start:end-grid.N_DOF_sol.get('p')] += face_imin.compute_momentum_laplace_SIP_penalty_term(self.settings.problem.type, RHS=True, g_int=g_int)
                        RHS[start:end-grid.N_DOF_sol.get('p')] += face_imin.compute_momentum_laplace_SIP_symmetrizing_term(self.settings.problem.type, RHS=True, g_int=g_int)
                        RHS[start:end-grid.N_DOF_sol.get('p')] += face_imin.compute_momentum_velocity_penalty_surface_integral(RHS=True, g_int=g_int)
                        if include_pressure_BC:
                            RHS[start:end-grid.N_DOF_sol.get('p')] += face_imin.compute_momentum_pressure_surface_integral(RHS=True, g_int=g_int)
                    if i==grid.Ni-1: # Dirichlet boundary
                        g_int = self.settings.problem.exact_solution_function(*xy_int['xy_int_imax'], self.settings.problem.type, 'solution')
                        RHS[end-grid.N_DOF_sol.get('p'):end] += face_imax.compute_continuity_surface_integral(RHS=True, g_int=g_int)
                        RHS[start:end-grid.N_DOF_sol.get('p')] += face_imax.compute_momentum_laplace_SIP_penalty_term(self.settings.problem.type, RHS=True, g_int=g_int)
                        RHS[start:end-grid.N_DOF_sol.get('p')] += face_imax.compute_momentum_laplace_SIP_symmetrizing_term(self.settings.problem.type, RHS=True, g_int=g_int)
                        RHS[start:end-grid.N_DOF_sol.get('p')] += face_imax.compute_momentum_velocity_penalty_surface_integral(RHS=True, g_int=g_int)
                        if include_pressure_BC:
                            RHS[start:end-grid.N_DOF_sol.get('p')] += face_imax.compute_momentum_pressure_surface_integral(RHS=True, g_int=g_int)

                if j==0: # Dirichlet boundary
                    g_int = self.settings.problem.exact_solution_function(*xy_int['xy_int_jmin'], self.settings.problem.type, 'solution')
                    RHS[end-grid.N_DOF_sol.get('p'):end] += face_jmin.compute_continuity_surface_integral(RHS=True, g_int=g_int)
                    RHS[start:end-grid.N_DOF_sol.get('p')] += face_jmin.compute_momentum_laplace_SIP_penalty_term(self.settings.problem.type, RHS=True, g_int=g_int)
                    RHS[start:end-grid.N_DOF_sol.get('p')] += face_jmin.compute_momentum_laplace_SIP_symmetrizing_term(self.settings.problem.type, RHS=True, g_int=g_int)
                    RHS[start:end-grid.N_DOF_sol.get('p')] += face_jmin.compute_momentum_velocity_penalty_surface_integral(RHS=True, g_int=g_int)
                    if include_pressure_BC:
                        RHS[start:end-grid.N_DOF_sol.get('p')] += face_jmin.compute_momentum_pressure_surface_integral(RHS=True, g_int=g_int)
                if j==grid.Nj-1: # Dirichlet boundary
                    g_int = self.settings.problem.exact_solution_function(*xy_int['xy_int_jmax'], self.settings.problem.type, 'solution')
                    RHS[end-grid.N_DOF_sol.get('p'):end] += face_jmax.compute_continuity_surface_integral(RHS=True, g_int=g_int)
                    RHS[start:end-grid.N_DOF_sol.get('p')] += face_jmax.compute_momentum_laplace_SIP_penalty_term(self.settings.problem.type, RHS=True, g_int=g_int)
                    RHS[start:end-grid.N_DOF_sol.get('p')] += face_jmax.compute_momentum_laplace_SIP_symmetrizing_term(self.settings.problem.type, RHS=True, g_int=g_int)
                    RHS[start:end-grid.N_DOF_sol.get('p')] += face_jmax.compute_momentum_velocity_penalty_surface_integral(RHS=True, g_int=g_int)
                    if include_pressure_BC:
                        RHS[start:end-grid.N_DOF_sol.get('p')] += face_jmax.compute_momentum_pressure_surface_integral(RHS=True, g_int=g_int)
        
        grid.RHS = RHS