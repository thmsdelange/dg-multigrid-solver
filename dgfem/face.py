import numpy as np

from dgfem.boundary import Boundary
from dgfem.interpolation import Interpolation

class Face:
    def __init__(self, element_L, element_R, direction, grid):
        self.element_L, self.element_R = element_L, element_R
        self.grid = grid
        self.settings = grid.settings
        self.interpol = Interpolation()

        if isinstance(self.element_L, Boundary):
            self.h_F = np.sqrt(element_R.A)
            self.J = self.element_R.gt.get('J').get(f'{direction}min')
            weights_L = self.element_R.weights
            weights_R = self.element_R.weights
            norms_L = self.element_R.norms
            norms_R = self.element_R.norms
        elif isinstance(self.element_R, Boundary):
            self.h_F = np.sqrt(element_L.A)
            self.J = self.element_L.gt.get('J').get(f'{direction}max')
            weights_L = self.element_L.weights
            weights_R = self.element_L.weights
            norms_L = self.element_L.norms
            norms_R = self.element_L.norms
        else:
            self.h_F = (np.sqrt(element_L.A) + np.sqrt(element_R.A))/2
            # self.J = self.element_R.gt.get('J').get(f'{direction}min')
            self.J = self.element_L.gt.get('J').get(f'{direction}max')
            # np.testing.assert_allclose(self.element_L.gt.get('J').get(f'{direction}max').get('u'), self.element_R.gt.get('J').get(f'{direction}min').get('u'), atol=1.e-15)
            weights_L = self.element_L.weights
            weights_R = self.element_R.weights
            norms_L = self.element_L.norms
            norms_R = self.element_R.norms

        self.nu = self.settings.problem.kinematic_viscosity
        
        if direction!='i' and direction!='j':
            raise NotImplementedError(f'Direction {direction} is not implemented, only i and j are possible arguments (2D implementation)')
        self.dir = direction

        if self.settings.problem.orthonormal_on_physical_element:
            # self.V_DOF_int_L, weights_L, norms_L = self.interpol.orthonormalize_Gram_Schmidt(getattr(self.grid, f'V_DOF_int_{self.dir}L'), self.J, self.grid.w_int)
            self.V_DOF_int_L = {'u': {'u': []}}
            self.Vr_DOF_int_L = {'u': {'u': []}}
            self.Vs_DOF_int_L = {'u': {'u': []}}
            self.V_DOF_int_R = {'u': {'u': []}}
            self.Vr_DOF_int_R = {'u': {'u': []}}
            self.Vs_DOF_int_R = {'u': {'u': []}}

            self.V_DOF_int_L['u']['u'] = np.array([getattr(self.grid, f'V_DOF_int_{self.dir}L').get('u').get('u') @ weights_L[:,j] for j in range(getattr(self.grid, f'V_DOF_int_{self.dir}L').get('u').get('u').shape[1])]).T * norms_L
            self.Vr_DOF_int_L['u']['u'] = np.array([getattr(self.grid, f'Vr_DOF_int_{self.dir}L').get('u').get('u') @ weights_L[:,j] for j in range(getattr(self.grid, f'Vr_DOF_int_{self.dir}L').get('u').get('u').shape[1])]).T * norms_L
            self.Vs_DOF_int_L['u']['u'] = np.array([getattr(self.grid, f'Vs_DOF_int_{self.dir}L').get('u').get('u') @ weights_L[:,j] for j in range(getattr(self.grid, f'Vs_DOF_int_{self.dir}L').get('u').get('u').shape[1])]).T * norms_L
            
            # self.V_DOF_int_R, weights_R, norms_R = self.interpol.orthonormalize_Gram_Schmidt(getattr(self.grid, f'V_DOF_int_{self.dir}R'), self.J, self.grid.w_int)
            self.V_DOF_int_R['u']['u'] = np.array([getattr(self.grid, f'V_DOF_int_{self.dir}R').get('u').get('u') @ weights_R[:,j] for j in range(getattr(self.grid, f'V_DOF_int_{self.dir}R').get('u').get('u').shape[1])]).T * norms_R
            self.Vr_DOF_int_R['u']['u'] = np.array([getattr(self.grid, f'Vr_DOF_int_{self.dir}R').get('u').get('u') @ weights_R[:,j] for j in range(getattr(self.grid, f'Vr_DOF_int_{self.dir}R').get('u').get('u').shape[1])]).T * norms_R
            self.Vs_DOF_int_R['u']['u'] = np.array([getattr(self.grid, f'Vs_DOF_int_{self.dir}R').get('u').get('u') @ weights_R[:,j] for j in range(getattr(self.grid, f'Vs_DOF_int_{self.dir}R').get('u').get('u').shape[1])]).T * norms_R
        else:
            self.V_DOF_int_L = getattr(self.grid, f'V_DOF_int_{self.dir}L')
            self.Vr_DOF_int_L = getattr(self.grid, f'Vr_DOF_int_{self.dir}L')
            self.Vs_DOF_int_L = getattr(self.grid, f'Vs_DOF_int_{self.dir}L')
            
            self.V_DOF_int_R = getattr(self.grid, f'V_DOF_int_{self.dir}R')
            self.Vr_DOF_int_R = getattr(self.grid, f'Vr_DOF_int_{self.dir}R')
            self.Vs_DOF_int_R = getattr(self.grid, f'Vs_DOF_int_{self.dir}R')

    def compute_continuity_MMS_u_dot_n_surface_integral(self, u_int):
        if isinstance(self.element_L, Boundary):
            return -(np.einsum('j,j,j,j->', u_int.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_R.gt.get('n').get(f'{self.dir}min').get('p')[:,0]) +\
                     np.einsum('j,j,j,j->', u_int.get('v').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_R.gt.get('n').get(f'{self.dir}min').get('p')[:,1]))    ### minus un because the normal is defined in positive r,s. So, the outward normal velocity is in the opposite direction for the left face
        elif isinstance(self.element_R, Boundary):
            return np.einsum('j,j,j,j->', u_int.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_L.gt.get('n').get(f'{self.dir}max').get('p')[:,0]) +\
                   np.einsum('j,j,j,j->', u_int.get('v').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_L.gt.get('n').get(f'{self.dir}max').get('p')[:,1])
        else:
            return None
        
    def compute_continuity_surface_integral(self, RHS=False, g_int=None):
        if isinstance(self.element_L, Boundary):
            if RHS:
                RHS = -(np.einsum('ji,j,j,j,j->i', self.V_DOF_int_R.get('p').get('p'), g_int.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_R.gt.get('n').get(f'{self.dir}min').get('p')[:,0]) +\
                        np.einsum('ji,j,j,j,j->i', self.V_DOF_int_R.get('p').get('p'), g_int.get('v').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_R.gt.get('n').get(f'{self.dir}min').get('p')[:,1]))
                return RHS
            else:
                C_Ru = -np.einsum('ji,j,j,j->ij', self.V_DOF_int_R.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_R.gt.get('n').get(f'{self.dir}min').get('p')[:,0])
                C_Rv = -np.einsum('ji,j,j,j->ij', self.V_DOF_int_R.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_R.gt.get('n').get(f'{self.dir}min').get('p')[:,1])
                C_Lu, C_Lv = np.zeros_like(C_Ru), np.zeros_like(C_Rv)
        elif isinstance(self.element_R, Boundary):
            if RHS:
                RHS = np.einsum('ji,j,j,j,j->i', self.V_DOF_int_L.get('p').get('p'), g_int.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_L.gt.get('n').get(f'{self.dir}max').get('p')[:,0]) +\
                      np.einsum('ji,j,j,j,j->i', self.V_DOF_int_L.get('p').get('p'), g_int.get('v').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_L.gt.get('n').get(f'{self.dir}max').get('p')[:,1])
                return RHS
            else:
                C_Lu = np.einsum('ji,j,j,j->ij', self.V_DOF_int_L.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_L.gt.get('n').get(f'{self.dir}max').get('p')[:,0])
                C_Lv = np.einsum('ji,j,j,j->ij', self.V_DOF_int_L.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_L.gt.get('n').get(f'{self.dir}max').get('p')[:,1])
                C_Ru, C_Rv = np.zeros_like(C_Lu), np.zeros_like(C_Lv)
        else:
            C_Lu = np.einsum('ji,j,j,j->ij', self.V_DOF_int_L.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_L.gt.get('n').get(f'{self.dir}max').get('p')[:,0])/2.
            C_Lv = np.einsum('ji,j,j,j->ij', self.V_DOF_int_L.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_L.gt.get('n').get(f'{self.dir}max').get('p')[:,1])/2.
            C_Ru = -np.einsum('ji,j,j,j->ij', self.V_DOF_int_R.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_R.gt.get('n').get(f'{self.dir}min').get('p')[:,0])/2.
            C_Rv = -np.einsum('ji,j,j,j->ij', self.V_DOF_int_R.get('u').get('p'), self.grid.w_int.get('p'), self.J.get('p'), self.element_R.gt.get('n').get(f'{self.dir}min').get('p')[:,1])/2.

        res_LLu, res_LLv = np.einsum('ij,jk->ki', C_Lu, self.V_DOF_int_L.get('p').get('p')), np.einsum('ij,jk->ki', C_Lv, self.V_DOF_int_L.get('p').get('p'))
        res_LRu, res_LRv = np.einsum('ij,jk->ki', C_Ru, self.V_DOF_int_L.get('p').get('p')), np.einsum('ij,jk->ki', C_Rv, self.V_DOF_int_L.get('p').get('p'))
        res_RLu, res_RLv = np.einsum('ij,jk->ki', C_Lu, self.V_DOF_int_R.get('p').get('p')), np.einsum('ij,jk->ki', C_Lv, self.V_DOF_int_R.get('p').get('p'))
        res_RRu, res_RRv = np.einsum('ij,jk->ki', C_Ru, self.V_DOF_int_R.get('p').get('p')), np.einsum('ij,jk->ki', C_Rv, self.V_DOF_int_R.get('p').get('p'))

        res_LL = np.concatenate((res_LLu, res_LLv), axis=1)
        res_LR = np.concatenate((res_LRu, res_LRv), axis=1)
        res_RL = np.concatenate((res_RLu, res_RLv), axis=1)
        res_RR = np.concatenate((res_RRu, res_RRv), axis=1)
        return res_LL, res_LR, res_RL, res_RR
    
    def compute_momentum_laplace_SIP_terms(self, problem):
        res_LL, res_LR, res_RL, res_RR = [
            sum(x) for x in zip(self.compute_momentum_laplace_SIP_flux_term(problem), self.compute_momentum_laplace_SIP_penalty_term(problem), self.compute_momentum_laplace_SIP_symmetrizing_term(problem))
        ]
        if not (np.count_nonzero(res_LL)==0 or np.count_nonzero(res_LR)==0 or np.count_nonzero(res_RL)==0 or np.count_nonzero(res_RR)==0):
            np.testing.assert_allclose(res_LL, res_LL.T, rtol=1e-6, atol=1e-2)
            np.testing.assert_allclose(res_RR, res_RR.T, rtol=1e-6, atol=1e-2)
            np.testing.assert_allclose(res_LR, res_RL.T, rtol=1e-6, atol=1e-2)
        elif np.count_nonzero(res_LR)==0:
            np.testing.assert_allclose(res_LL, res_LL.T, rtol=1e-6, atol=1e-2)
        elif np.count_nonzero(res_RL)==0:
            np.testing.assert_allclose(res_RR, res_RR.T, rtol=1e-6, atol=1e-2)
        return res_LL, res_LR, res_RL, res_RR

    def compute_momentum_laplace_SIP_flux_term(self, problem):
        if isinstance(self.element_L, Boundary): 
            ux_R = np.einsum('ji,j->ij', self.Vr_DOF_int_R.get('u').get('u'), self.element_R.gt.get('rx').get(f'{self.dir}min').get('u')) +\
                   np.einsum('ji,j->ij', self.Vs_DOF_int_R.get('u').get('u'), self.element_R.gt.get('sx').get(f'{self.dir}min').get('u'))
            uy_R = np.einsum('ji,j->ij', self.Vr_DOF_int_R.get('u').get('u'), self.element_R.gt.get('ry').get(f'{self.dir}min').get('u')) +\
                   np.einsum('ji,j->ij', self.Vs_DOF_int_R.get('u').get('u'), self.element_R.gt.get('sy').get(f'{self.dir}min').get('u'))
            un_R = self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0] * ux_R + self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1] * uy_R
            un_L = np.zeros_like(un_R)
        
            C_R = self.nu*un_R*self.grid.w_int.get('u')*self.J.get('u')
            C_L = np.zeros_like(C_R)
        elif isinstance(self.element_R, Boundary):
            ux_L = np.einsum('ji,j->ij', self.Vr_DOF_int_L.get('u').get('u'), self.element_L.gt.get('rx').get(f'{self.dir}max').get('u')) +\
                   np.einsum('ji,j->ij', self.Vs_DOF_int_L.get('u').get('u'), self.element_L.gt.get('sx').get(f'{self.dir}max').get('u'))
            uy_L = np.einsum('ji,j->ij', self.Vr_DOF_int_L.get('u').get('u'), self.element_L.gt.get('ry').get(f'{self.dir}max').get('u')) +\
                   np.einsum('ji,j->ij', self.Vs_DOF_int_L.get('u').get('u'), self.element_L.gt.get('sy').get(f'{self.dir}max').get('u'))
            un_L = self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0] * ux_L + self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1] * uy_L
            un_R = np.zeros_like(un_L)

            C_L = self.nu*un_L*self.grid.w_int.get('u')*self.J.get('u')
            C_R = np.zeros_like(C_L)
        else:  
            ux_L = np.einsum('ji,j->ij', self.Vr_DOF_int_L.get('u').get('u'), self.element_L.gt.get('rx').get(f'{self.dir}max').get('u')) +\
                   np.einsum('ji,j->ij', self.Vs_DOF_int_L.get('u').get('u'), self.element_L.gt.get('sx').get(f'{self.dir}max').get('u'))
            uy_L = np.einsum('ji,j->ij', self.Vr_DOF_int_L.get('u').get('u'), self.element_L.gt.get('ry').get(f'{self.dir}max').get('u')) +\
                   np.einsum('ji,j->ij', self.Vs_DOF_int_L.get('u').get('u'), self.element_L.gt.get('sy').get(f'{self.dir}max').get('u'))
            ux_R = np.einsum('ji,j->ij', self.Vr_DOF_int_R.get('u').get('u'), self.element_R.gt.get('rx').get(f'{self.dir}min').get('u')) +\
                   np.einsum('ji,j->ij', self.Vs_DOF_int_R.get('u').get('u'), self.element_R.gt.get('sx').get(f'{self.dir}min').get('u'))
            uy_R = np.einsum('ji,j->ij', self.Vr_DOF_int_R.get('u').get('u'), self.element_R.gt.get('ry').get(f'{self.dir}min').get('u')) +\
                   np.einsum('ji,j->ij', self.Vs_DOF_int_R.get('u').get('u'), self.element_R.gt.get('sy').get(f'{self.dir}min').get('u'))

            un_L = self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0] * ux_L + self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1] * uy_L
            un_R = self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0] * ux_R + self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1] * uy_R

            C_L = self.nu/2.*un_L*self.grid.w_int.get('u')*self.J.get('u')
            C_R = self.nu/2.*un_R*self.grid.w_int.get('u')*self.J.get('u')

        res_LL = -np.einsum('ij,jk->ki', C_L, self.V_DOF_int_L.get('u').get('u'))
        res_LR = -np.einsum('ij,jk->ki', C_R, self.V_DOF_int_L.get('u').get('u'))
        res_RL = np.einsum('ij,jk->ki', C_L, self.V_DOF_int_R.get('u').get('u'))
        res_RR = np.einsum('ij,jk->ki', C_R, self.V_DOF_int_R.get('u').get('u'))
        
        if problem=='Poisson':
            return res_LL, res_LR, res_RL, res_RR
        elif problem=='Stokes':
            res_LL = np.concatenate((np.concatenate((res_LL, np.zeros_like(res_LL)), axis=1), np.concatenate((np.zeros_like(res_LL), res_LL), axis=1)), axis=0)
            res_LR = np.concatenate((np.concatenate((res_LR, np.zeros_like(res_LR)), axis=1), np.concatenate((np.zeros_like(res_LR), res_LR), axis=1)), axis=0)
            res_RL = np.concatenate((np.concatenate((res_RL, np.zeros_like(res_RL)), axis=1), np.concatenate((np.zeros_like(res_RL), res_RL), axis=1)), axis=0)
            res_RR = np.concatenate((np.concatenate((res_RR, np.zeros_like(res_RR)), axis=1), np.concatenate((np.zeros_like(res_RR), res_RR), axis=1)), axis=0)
            return res_LL, res_LR, res_RL, res_RR

    def compute_momentum_laplace_SIP_penalty_term(self, problem, RHS=False, g_int=None):
        if isinstance(self.element_L, Boundary):
            if RHS:
                RHS_u = self.grid.sigma*self.nu/self.h_F*np.einsum('ji,j,j,j->i', self.V_DOF_int_R.get('u').get('u'), g_int.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                if problem=='Poisson':
                    return RHS_u
                elif problem=='Stokes':
                    RHS_v = self.grid.sigma*self.nu/self.h_F*np.einsum('ji,j,j,j->i', self.V_DOF_int_R.get('u').get('u'), g_int.get('v').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                    return np.concatenate((RHS_u, RHS_v), axis=0)
            else:
                C_R = -self.grid.sigma*self.nu/self.h_F*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                C_L = np.zeros_like(C_R)
        elif isinstance(self.element_R, Boundary):
            if RHS:
                RHS_u = self.grid.sigma*self.nu/self.h_F*np.einsum('ji,j,j,j->i', self.V_DOF_int_L.get('u').get('u'), g_int.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                if problem=='Poisson':
                    return RHS_u
                elif problem=='Stokes':
                    RHS_v = self.grid.sigma*self.nu/self.h_F*np.einsum('ji,j,j,j->i', self.V_DOF_int_L.get('u').get('u'), g_int.get('v').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                    return np.concatenate((RHS_u, RHS_v), axis=0)
            else:
                C_L = self.grid.sigma*self.nu/self.h_F*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                C_R = np.zeros_like(C_L)
        else:
            C_L = self.grid.sigma*self.nu/self.h_F*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
            C_R = -self.grid.sigma*self.nu/self.h_F*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))

        res_LL = np.einsum('ij,jk->ki', C_L,  self.V_DOF_int_L.get('u').get('u'))
        res_LR = np.einsum('ij,jk->ki', C_R,  self.V_DOF_int_L.get('u').get('u'))
        res_RL = -np.einsum('ij,jk->ki', C_L, self.V_DOF_int_R.get('u').get('u'))
        res_RR = -np.einsum('ij,jk->ki', C_R, self.V_DOF_int_R.get('u').get('u'))
        
        if problem=='Poisson':
            return res_LL, res_LR, res_RL, res_RR
        elif problem=='Stokes':
            res_LL = np.concatenate((np.concatenate((res_LL, np.zeros_like(res_LL)), axis=1), np.concatenate((np.zeros_like(res_LL), res_LL), axis=1)), axis=0)
            res_LR = np.concatenate((np.concatenate((res_LR, np.zeros_like(res_LR)), axis=1), np.concatenate((np.zeros_like(res_LR), res_LR), axis=1)), axis=0)
            res_RL = np.concatenate((np.concatenate((res_RL, np.zeros_like(res_RL)), axis=1), np.concatenate((np.zeros_like(res_RL), res_RL), axis=1)), axis=0)
            res_RR = np.concatenate((np.concatenate((res_RR, np.zeros_like(res_RR)), axis=1), np.concatenate((np.zeros_like(res_RR), res_RR), axis=1)), axis=0)
            return res_LL, res_LR, res_RL, res_RR
    
    def compute_momentum_laplace_SIP_symmetrizing_term(self, problem, RHS=False, g_int=None):
        if isinstance(self.element_L, Boundary):    
            psix_R = np.einsum('ji,j->ij', self.Vr_DOF_int_R.get('u').get('u'), self.element_R.gt.get('rx').get(f'{self.dir}min').get('u')) + np.einsum('ji,j->ij', self.Vs_DOF_int_R.get('u').get('u'), self.element_R.gt.get('sx').get(f'{self.dir}min').get('u'))
            psiy_R = np.einsum('ji,j->ij', self.Vr_DOF_int_R.get('u').get('u'), self.element_R.gt.get('ry').get(f'{self.dir}min').get('u')) + np.einsum('ji,j->ij', self.Vs_DOF_int_R.get('u').get('u'), self.element_R.gt.get('sy').get(f'{self.dir}min').get('u'))
            
            psin_R = self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0] * psix_R + self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1] * psiy_R
            psin_L = np.zeros_like(psin_R)

            if RHS:
                RHS_u = self.nu*np.einsum('ji,i,i,i->j', psin_R, g_int.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                if problem=='Poisson':
                    return RHS_u
                elif problem=='Stokes':
                    RHS_v = self.nu*np.einsum('ji,i,i,i->j', psin_R, g_int.get('v').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                    return np.concatenate((RHS_u, RHS_v), axis=0)
            else:
                C_R = -self.nu*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                C_L = np.zeros_like(C_R)
        elif isinstance(self.element_R, Boundary):
            psix_L = np.einsum('ji,j->ij', self.Vr_DOF_int_L.get('u').get('u'), self.element_L.gt.get('rx').get(f'{self.dir}max').get('u')) + np.einsum('ji,j->ij', self.Vs_DOF_int_L.get('u').get('u'), self.element_L.gt.get('sx').get(f'{self.dir}max').get('u'))
            psiy_L = np.einsum('ji,j->ij', self.Vr_DOF_int_L.get('u').get('u'), self.element_L.gt.get('ry').get(f'{self.dir}max').get('u')) + np.einsum('ji,j->ij', self.Vs_DOF_int_L.get('u').get('u'), self.element_L.gt.get('sy').get(f'{self.dir}max').get('u'))

            psin_L = self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0] * psix_L + self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1] * psiy_L
            psin_R = np.zeros_like(psin_L)
            if RHS:
                RHS_u = -self.nu*np.einsum('ji,i,i,i->j', psin_L, g_int.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                if problem=='Poisson':
                    return RHS_u
                elif problem=='Stokes':
                    RHS_v = -self.nu*np.einsum('ji,i,i,i->j', psin_L, g_int.get('v').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                    return np.concatenate((RHS_u, RHS_v), axis=0)
            else:
                C_L = self.nu*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
                C_R = np.zeros_like(C_L)
        else:
            psix_L = np.einsum('ji,j->ij', self.Vr_DOF_int_L.get('u').get('u'), self.element_L.gt.get('rx').get(f'{self.dir}max').get('u')) + np.einsum('ji,j->ij', self.Vs_DOF_int_L.get('u').get('u'), self.element_L.gt.get('sx').get(f'{self.dir}max').get('u'))
            psiy_L = np.einsum('ji,j->ij', self.Vr_DOF_int_L.get('u').get('u'), self.element_L.gt.get('ry').get(f'{self.dir}max').get('u')) + np.einsum('ji,j->ij', self.Vs_DOF_int_L.get('u').get('u'), self.element_L.gt.get('sy').get(f'{self.dir}max').get('u'))

            psix_R = np.einsum('ji,j->ij', self.Vr_DOF_int_R.get('u').get('u'), self.element_R.gt.get('rx').get(f'{self.dir}min').get('u')) + np.einsum('ji,j->ij', self.Vs_DOF_int_R.get('u').get('u'), self.element_R.gt.get('sx').get(f'{self.dir}min').get('u'))
            psiy_R = np.einsum('ji,j->ij', self.Vr_DOF_int_R.get('u').get('u'), self.element_R.gt.get('ry').get(f'{self.dir}min').get('u')) + np.einsum('ji,j->ij', self.Vs_DOF_int_R.get('u').get('u'), self.element_R.gt.get('sy').get(f'{self.dir}min').get('u'))
            
            psin_L = self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0] * psix_L + self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1] * psiy_L
            psin_R = self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0] * psix_R + self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1] * psiy_R

            C_L = self.nu/2.*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
            C_R = -self.nu/2.*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))
        
        res_LL = -np.einsum('ij,kj->ki', C_L, psin_L)
        res_LR = -np.einsum('ij,kj->ki', C_R, psin_L)
        res_RL = -np.einsum('ij,kj->ki', C_L, psin_R)
        res_RR = -np.einsum('ij,kj->ki', C_R, psin_R)
        
        if problem=='Poisson':
            return res_LL, res_LR, res_RL, res_RR
        elif problem=='Stokes':
            res_LL = np.concatenate((np.concatenate((res_LL, np.zeros_like(res_LL)), axis=1), np.concatenate((np.zeros_like(res_LL), res_LL), axis=1)), axis=0)
            res_LR = np.concatenate((np.concatenate((res_LR, np.zeros_like(res_LR)), axis=1), np.concatenate((np.zeros_like(res_LR), res_LR), axis=1)), axis=0)
            res_RL = np.concatenate((np.concatenate((res_RL, np.zeros_like(res_RL)), axis=1), np.concatenate((np.zeros_like(res_RL), res_RL), axis=1)), axis=0)
            res_RR = np.concatenate((np.concatenate((res_RR, np.zeros_like(res_RR)), axis=1), np.concatenate((np.zeros_like(res_RR), res_RR), axis=1)), axis=0)
            return res_LL, res_LR, res_RL, res_RR
    
    def compute_momentum_pressure_surface_integral(self, RHS=False, g_int=None): ### CHECK IF INDICES CORRECT!
        if isinstance(self.element_L, Boundary):
            if RHS:
                gx = g_int.get('p').get('u')*self.grid.w_int.get('u')*self.J.get('u')*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0]
                gy = g_int.get('p').get('u')*self.grid.w_int.get('u')*self.J.get('u')*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]
                RHS_u = -np.einsum('ji,j->i', self.V_DOF_int_R.get('u').get('u'), gx)
                RHS_v = -np.einsum('ji,j->i', self.V_DOF_int_R.get('u').get('u'), gy)
                return np.concatenate((RHS_u, RHS_v), axis=0)
            else:
                C_Rx = np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('p').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0]
                C_Ry = np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('p').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]
                C_Lx, C_Ly = np.zeros_like(C_Rx), np.zeros_like(C_Ry)
        elif isinstance(self.element_R, Boundary):
            if RHS:
                gx = g_int.get('p').get('u')*self.grid.w_int.get('u')*self.J.get('u')*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0]
                gy = g_int.get('p').get('u')*self.grid.w_int.get('u')*self.J.get('u')*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]
                RHS_u = np.einsum('ji,j->i', self.V_DOF_int_L.get('u').get('u'), gx)
                RHS_v = np.einsum('ji,j->i', self.V_DOF_int_L.get('u').get('u'), gy)
                return np.concatenate((RHS_u, RHS_v), axis=0)
            else:
                C_Lx = np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('p').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0]
                C_Ly = np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('p').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]
                C_Rx, C_Ry = np.zeros_like(C_Lx), np.zeros_like(C_Ly)
        else:
            C_Lx = np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('p').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0]/2.
            C_Ly = np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('p').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]/2.
            C_Rx = np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('p').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0]/2.
            C_Ry = np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('p').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]/2.

        res_LLx, res_LLy = np.einsum('ij,jk->ki', C_Lx, self.V_DOF_int_L.get('u').get('u')), np.einsum('ij,jk->ki', C_Ly, self.V_DOF_int_L.get('u').get('u'))
        res_LRx, res_LRy = np.einsum('ij,jk->ki', C_Rx, self.V_DOF_int_L.get('u').get('u')), np.einsum('ij,jk->ki', C_Ry, self.V_DOF_int_L.get('u').get('u'))
        res_RLx, res_RLy = -np.einsum('ij,jk->ki', C_Lx, self.V_DOF_int_R.get('u').get('u')), -np.einsum('ij,jk->ki', C_Ly, self.V_DOF_int_R.get('u').get('u'))
        res_RRx, res_RRy = -np.einsum('ij,jk->ki', C_Rx, self.V_DOF_int_R.get('u').get('u')), -np.einsum('ij,jk->ki', C_Ry, self.V_DOF_int_R.get('u').get('u'))

        res_LL = np.concatenate((res_LLx, res_LLy), axis=0)
        res_LR = np.concatenate((res_LRx, res_LRy), axis=0)
        res_RL = np.concatenate((res_RLx, res_RLy), axis=0)
        res_RR = np.concatenate((res_RRx, res_RRy), axis=0)
        return res_LL, res_LR, res_RL, res_RR
    
    def compute_momentum_velocity_penalty_surface_integral(self, RHS=False, g_int=None):
        if isinstance(self.element_L, Boundary):
            if RHS:
                gn = g_int.get('u').get('u')*self.grid.w_int.get('u')*self.J.get('u')*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0] +\
                     g_int.get('v').get('u')*self.grid.w_int.get('u')*self.J.get('u')*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]
                RHS_u = self.grid.gamma*np.einsum('ji,j,j->i', self.V_DOF_int_R.get('u').get('u'), gn, self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0])/self.h_F
                RHS_v = self.grid.gamma*np.einsum('ji,j,j->i', self.V_DOF_int_R.get('u').get('u'), gn, self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1])/self.h_F
                return np.concatenate((RHS_u, RHS_v), axis=0)
            else:
                C_Rux = -self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0]/self.h_F*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0]
                C_Ruy = -self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0]/self.h_F*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]
                C_Rvx = -self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]/self.h_F*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0]
                C_Rvy = -self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]/self.h_F*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]
                C_Lux, C_Luy, C_Lvx, C_Lvy = np.zeros_like(C_Rux), np.zeros_like(C_Ruy), np.zeros_like(C_Rvx), np.zeros_like(C_Rvy)
        elif isinstance(self.element_R, Boundary):
            if RHS:
                gn = g_int.get('u').get('u')*self.grid.w_int.get('u')*self.J.get('u')*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0] +\
                     g_int.get('v').get('u')*self.grid.w_int.get('u')*self.J.get('u')*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]
                RHS_u = self.grid.gamma*np.einsum('ji,j,j->i', self.V_DOF_int_L.get('u').get('u'), gn, self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0])/self.h_F
                RHS_v = self.grid.gamma*np.einsum('ji,j,j->i', self.V_DOF_int_L.get('u').get('u'), gn, self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1])/self.h_F
                return np.concatenate((RHS_u, RHS_v), axis=0)
            else:
                C_Lux = self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0]/self.h_F*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0]
                C_Luy = self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0]/self.h_F*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]
                C_Lvx = self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]/self.h_F*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0]
                C_Lvy = self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]/self.h_F*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]
                C_Rux, C_Ruy, C_Rvx, C_Rvy = np.zeros_like(C_Lux), np.zeros_like(C_Luy), np.zeros_like(C_Lvx), np.zeros_like(C_Lvy)
        else:
            C_Lux = self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0]/self.h_F*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0]
            C_Luy = self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0]/self.h_F*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]
            C_Lvx = self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]/self.h_F*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,0]
            C_Lvy = self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_L.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]/self.h_F*self.element_L.gt.get('n').get(f'{self.dir}max').get('u')[:,1]
            C_Rux = -self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0]/self.h_F*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0]
            C_Ruy = -self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0]/self.h_F*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]
            C_Rvx = -self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]/self.h_F*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,0]
            C_Rvy = -self.grid.gamma*np.einsum('ji,j,j->ij', self.V_DOF_int_R.get('u').get('u'), self.grid.w_int.get('u'), self.J.get('u'))*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]/self.h_F*self.element_R.gt.get('n').get(f'{self.dir}min').get('u')[:,1]

        res_LLux, res_LLvx = np.einsum('ij,jk->ki', C_Lux, self.V_DOF_int_L.get('u').get('u')), np.einsum('ij,jk->ki', C_Lvx, self.V_DOF_int_L.get('u').get('u'))
        res_LLuy, res_LLvy = np.einsum('ij,jk->ki', C_Luy, self.V_DOF_int_L.get('u').get('u')), np.einsum('ij,jk->ki', C_Lvy, self.V_DOF_int_L.get('u').get('u'))
        res_LRux, res_LRvx = np.einsum('ij,jk->ki', C_Rux, self.V_DOF_int_L.get('u').get('u')), np.einsum('ij,jk->ki', C_Rvx, self.V_DOF_int_L.get('u').get('u'))
        res_LRuy, res_LRvy = np.einsum('ij,jk->ki', C_Ruy, self.V_DOF_int_L.get('u').get('u')), np.einsum('ij,jk->ki', C_Rvy, self.V_DOF_int_L.get('u').get('u'))
        res_RLux, res_RLvx = -np.einsum('ij,jk->ki', C_Lux, self.V_DOF_int_R.get('u').get('u')), -np.einsum('ij,jk->ki', C_Lvx, self.V_DOF_int_R.get('u').get('u'))
        res_RLuy, res_RLvy = -np.einsum('ij,jk->ki', C_Luy, self.V_DOF_int_R.get('u').get('u')), -np.einsum('ij,jk->ki', C_Lvy, self.V_DOF_int_R.get('u').get('u'))
        res_RRux, res_RRvx = -np.einsum('ij,jk->ki', C_Rux, self.V_DOF_int_R.get('u').get('u')), -np.einsum('ij,jk->ki', C_Rvx, self.V_DOF_int_R.get('u').get('u'))
        res_RRuy, res_RRvy = -np.einsum('ij,jk->ki', C_Ruy, self.V_DOF_int_R.get('u').get('u')), -np.einsum('ij,jk->ki', C_Rvy, self.V_DOF_int_R.get('u').get('u'))

        res_LL = np.concatenate((np.concatenate((res_LLux, res_LLvx), axis=1), np.concatenate((res_LLuy, res_LLvy), axis=1)), axis=0)
        res_LR = np.concatenate((np.concatenate((res_LRux, res_LRvx), axis=1), np.concatenate((res_LRuy, res_LRvy), axis=1)), axis=0)
        res_RL = np.concatenate((np.concatenate((res_RLux, res_RLvx), axis=1), np.concatenate((res_RLuy, res_RLvy), axis=1)), axis=0)
        res_RR = np.concatenate((np.concatenate((res_RRux, res_RRvx), axis=1), np.concatenate((res_RRuy, res_RRvy), axis=1)), axis=0)
        return res_LL, res_LR, res_RL, res_RR