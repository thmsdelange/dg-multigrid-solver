import numpy as np

from dgfem.interpolation import Interpolation
from utils.helpers import convert_to_dict

class Element:
    def __init__(self, x, y, grid, geometric_terms=None, xy_int=None):
        self.x, self.y = x, y
        self.grid = grid
        self.settings = grid.settings
        self.xy_int = xy_int
        self.gt = geometric_terms
        self.inv_mass_matrix = None
        self.interpol = Interpolation()

        if not isinstance(self.xy_int, dict):
            self.xy_int = {'xy_int': self.metric_xy_rs(self.x, self.y, self.grid.r_int, self.grid.r_int),
                           'xy_int_imin': self.metric_xy_rs(self.x, self.y, [-1], self.grid.r_int),
                           'xy_int_imax': self.metric_xy_rs(self.x, self.y, [1], self.grid.r_int),
                           'xy_int_jmin': self.metric_xy_rs(self.x, self.y, self.grid.r_int, [-1]),
                           'xy_int_jmax': self.metric_xy_rs(self.x, self.y, self.grid.r_int, [1]),
                           }
        if not isinstance(self.gt, dict):
            self.gt = {'J': {}, 'rx': {}, 'sx': {}, 'ry': {}, 'sy': {}, 'n': {}}
            self.gt['J']['e'], self.gt['rx']['e'], self.gt['sx']['e'], self.gt['ry']['e'], self.gt['sy']['e'] = self.compute_geometric_terms(self.x, self.y)
            self.gt['J']['imin'], self.gt['rx']['imin'], self.gt['sx']['imin'], self.gt['ry']['imin'], self.gt['sy']['imin'], self.gt['n']['imin'] = self.compute_geometric_terms(self.x, self.y, face='imin')
            self.gt['J']['imax'], self.gt['rx']['imax'], self.gt['sx']['imax'], self.gt['ry']['imax'], self.gt['sy']['imax'], self.gt['n']['imax'] = self.compute_geometric_terms(self.x, self.y, face='imax')
            self.gt['J']['jmin'], self.gt['rx']['jmin'], self.gt['sx']['jmin'], self.gt['ry']['jmin'], self.gt['sy']['jmin'], self.gt['n']['jmin'] = self.compute_geometric_terms(self.x, self.y, face='jmin')
            self.gt['J']['jmax'], self.gt['rx']['jmax'], self.gt['sx']['jmax'], self.gt['ry']['jmax'], self.gt['sy']['jmax'], self.gt['n']['jmax'] = self.compute_geometric_terms(self.x, self.y, face='jmax')
        self.A = np.einsum('ij,ij->', self.gt.get('J').get('e').get('u'), self.grid.w_int_2D.get('u'))
        
        ### not yet transformed to multiple variables u,v,p:
        if self.settings.problem.orthonormal_on_physical_element:
            self.V_DOF_int = {'u': {'u': []}}
            self.Vr_DOF_int = {'u': {'u': []}}
            self.Vs_DOF_int = {'u': {'u': []}}
            self.V_DOF_grid = {'u': {'u': []}}
            self.V_DOF_int['u']['u'], self.weights, self.norms = self.interpol.orthonormalize_Gram_Schmidt(self.grid.V_DOF_int.get('u').get('u'), self.gt.get('J').get('e').get('u'), self.grid.w_int_2D.get('u'))
            # V_int_DOF_test = np.array([self.grid.V_DOF_int @ weights[:,j] for j in range(self.grid.V_DOF_int.shape[1])]).T * norms
            # np.testing.assert_allclose(self.V_DOF_int, V_int_DOF_test)
            self.Vr_DOF_int['u']['u'] = np.array([self.grid.Vr_DOF_int.get('u').get('u') @ self.weights[:,j] for j in range(self.grid.Vr_DOF_int.get('u').get('u').shape[1])]).T * self.norms
            self.Vs_DOF_int['u']['u'] = np.array([self.grid.Vs_DOF_int.get('u').get('u') @ self.weights[:,j] for j in range(self.grid.Vs_DOF_int.get('u').get('u').shape[1])]).T * self.norms
            self.V_DOF_grid['u']['u'] = np.array([self.grid.V_DOF_grid.get('u').get('u') @ self.weights[:,j] for j in range(self.grid.V_DOF_grid.get('u').get('u').shape[1])]).T * self.norms
        else:
            self.weights = None
            self.norms = None
            self.V_DOF_int = self.grid.V_DOF_int
            self.Vr_DOF_int = self.grid.Vr_DOF_int
            self.Vs_DOF_int = self.grid.Vs_DOF_int
            self.V_DOF_grid = self.grid.V_DOF_grid
        
    def compute_geometric_terms(self, x, y, r_int=None, s_int=None, face=None, coarsening_factor=None, no_dict=False, vars=None):
        """Calculate geometric factors in integration nodes"""
        self.x_rs, self.y_rs = self.metric_xy_rs(x, y, self.grid.r_grid, self.grid.r_grid)
        
        if not r_int and not s_int:
            r_int, s_int = self.grid.r_int, self.grid.r_int
            Vr_grid_int = getattr(self.grid, 'Vr_grid_int_' + face if face else 'Vr_grid_int')
            Vs_grid_int = getattr(self.grid, 'Vs_grid_int_' + face if face else 'Vs_grid_int')
        else:
            if not face:
                Vr_grid_int, Vs_grid_int = self.interpol.grad_vandermonde2D(self.grid.N_grid, r_int, s_int)  # changed all these from self.grid.N_sol to self.grid.N_grid Vr/s_int_DOF -> Vr/s_int
            elif face=='imin':
                Vr_grid_int, Vs_grid_int = self.interpol.grad_vandermonde2D(self.grid.N_grid, [-1], s_int)
            elif face=='imax':
                Vr_grid_int, Vs_grid_int = self.interpol.grad_vandermonde2D(self.grid.N_grid, [1], s_int)
            elif face=='jmin':
                Vr_grid_int, Vs_grid_int = self.interpol.grad_vandermonde2D(self.grid.N_grid, r_int, [-1])
            elif face=='jmax':
                Vr_grid_int, Vs_grid_int = self.interpol.grad_vandermonde2D(self.grid.N_grid, r_int, [1])

        J_e_dict, J_f_dict, rx_dict, sx_dict, ry_dict, sy_dict, n_dict = {}, {}, {}, {}, {}, {}, {}
        _, r_int_dict, s_int_dict = convert_to_dict(None, r_int, s_int)
        keys = Vr_grid_int.get('u').keys() if not vars else vars
        for key in keys:
            Dr = (np.linalg.inv(self.grid.V_grid_grid.get('u').get('u')).T @ Vr_grid_int.get('u').get(key).T).T
            Ds = (np.linalg.inv(self.grid.V_grid_grid.get('u').get('u')).T @ Vs_grid_int.get('u').get(key).T).T

            xr, xs = Dr @ np.ravel(self.x_rs.get('u'), order='F'), Ds @ np.ravel(self.x_rs.get('u'), order='F')
            yr, ys = Dr @ np.ravel(self.y_rs.get('u'), order='F'), Ds @ np.ravel(self.y_rs.get('u'), order='F')
            if coarsening_factor: 
                xr *= coarsening_factor            
                xs *= coarsening_factor            
                yr *= coarsening_factor            
                ys *= coarsening_factor    
            if not face and (isinstance(r_int_dict[key], np.ndarray) and isinstance(s_int_dict[key], np.ndarray)):
                xr = xr.reshape((len(r_int_dict[key]), len(s_int_dict[key])), order='F')
                xs = xs.reshape((len(r_int_dict[key]), len(s_int_dict[key])), order='F')
                yr = yr.reshape((len(r_int_dict[key]), len(s_int_dict[key])), order='F')
                ys = ys.reshape((len(r_int_dict[key]), len(s_int_dict[key])), order='F')        
            
            
            J_e_dict[key] = xr*ys - yr*xs
            rx_dict[key], sx_dict[key] = ys/J_e_dict[key], -yr/J_e_dict[key]
            ry_dict[key], sy_dict[key] = -xs/J_e_dict[key], xr/J_e_dict[key]
            if face:
                if face=='imin' or face=='imax':
                    J_f_dict[key] = np.sqrt(xs**2 + ys**2)
                    n_dict[key] = (np.array([rx_dict[key], ry_dict[key]])/np.sqrt(rx_dict[key]**2+ry_dict[key]**2)).T
                elif face=='jmin' or face=='jmax':
                    J_f_dict[key] = np.sqrt(xr**2 + yr**2)
                    n_dict[key] = (np.array([sx_dict[key], sy_dict[key]])/np.sqrt(sx_dict[key]**2+sy_dict[key]**2)).T
        if not no_dict:
            if not face:
                return J_e_dict, rx_dict, sx_dict, ry_dict, sy_dict
            else:
                return J_f_dict, rx_dict, sx_dict, ry_dict, sy_dict, n_dict
        else:
            assert len(keys)==1
            if not face:
                return J_e_dict.get(key), rx_dict.get(key), sx_dict.get(key), ry_dict.get(key), sy_dict.get(key)
            else:
                return J_f_dict.get(key), rx_dict.get(key), sx_dict.get(key), ry_dict.get(key), sy_dict.get(key), n_dict.get(key)

    def metric_xy_rs(self, x, y, r, s, no_dict=False, vars=None):
        V = self.interpol.vandermonde2D(self.grid.N_grid, r, s)
        _, r_dict, s_dict = convert_to_dict(None, r, s)

        x_rs_dict, y_rs_dict = {}, {}
        keys = V.get('u').keys() if not vars else vars
        for key in keys:
            li = (np.linalg.inv(self.grid.V_grid_grid.get('u').get('u').T) @ V['u'][key].T).T

            x_rs, y_rs = li @ np.ravel(x, order='F'), li @ np.ravel(y, order='F')
            x_rs_dict[key], y_rs_dict[key] = x_rs.reshape((len(r_dict[key]),len(s_dict[key])), order='F'), y_rs.reshape((len(r_dict[key]),len(s_dict[key])), order='F')
        if not no_dict:
            return x_rs_dict, y_rs_dict
        else:
            assert len(keys)==1
            return x_rs_dict.get(key), y_rs_dict.get(key)
    
    def compute_mass_matrix(self):
        return np.einsum('ij,jk->ik', np.einsum('ji,j,j->ij', self.V_DOF_int.get('u').get('u'), np.ravel(self.grid.w_int_2D.get('u'), order='F'), np.ravel(self.gt.get('J').get('e').get('u'), order='F')), self.V_DOF_int.get('u').get('u'))
    
    def orthonormalize_Gram_Schmidt_old(self, V_DOF_int, w_int, mass_matrix):
        """Orthonormalize the basis on the physical space using the Gram-Schmidt orthonormalisation technique"""
        weights = np.zeros((V_DOF_int.shape[1],V_DOF_int.shape[1]))
        for j in range(V_DOF_int.shape[1]):
            weights[j,j] = 1.
            for i in range(j):
                weights[i,j] = -mass_matrix[j,i]/mass_matrix[i,i]

        V_int_DOF_orthogonal = V_DOF_int @ weights
        norm = np.zeros(V_int_DOF_orthogonal.shape[1])
        for j in range(V_int_DOF_orthogonal.shape[1]):
            norm[j] = 1./np.sqrt(V_int_DOF_orthogonal[:,j] * V_int_DOF_orthogonal[:,j] * np.ravel(self.gt.get('J').get('e'), order='F') @ np.ravel(w_int, order='F'))

        V_int_DOF_orthonormal = V_int_DOF_orthogonal*norm
        return V_int_DOF_orthonormal, weights, norm
    
    def compute_pressure_integral(self, p):
        p_int = self.grid.V_DOF_int.get('p').get('p') @ p
        return np.einsum('j,j,j->', p_int, np.ravel(self.gt.get('J').get('e').get('p'), order='F'), np.ravel(self.grid.w_int_2D.get('p'), order='F'))
    
    def compute_continuity_MMS_volume_integral(self, f_int):
        return np.einsum('ij,ij,ij->', f_int.get('p'), self.grid.w_int_2D.get('p'), self.gt.get('J').get('e').get('p'))
    
    def compute_source_continuity_volume_integral(self, f_int): ### minus sign for consistency with LHS
        return -np.einsum('ji,j,j,j->i', self.V_DOF_int.get('p').get('p'), np.ravel(self.grid.w_int_2D.get('p'), order='F'), np.ravel(self.gt.get('J').get('e').get('p'), order='F'), np.ravel(f_int.get('p'), order='F'))
    
    def compute_source_momentum_volume_integral(self, problem, f_int):
        res_x = np.einsum('ji,j,j,j->i', self.V_DOF_int.get('u').get('u'), np.ravel(self.grid.w_int_2D.get('u'), order='F'), np.ravel(self.gt.get('J').get('e').get('u'), order='F'), np.ravel(f_int[0].get('u'), order='F'))
        if problem=='Poisson':
            return res_x
        elif problem=='Stokes':
            res_y = np.einsum('ji,j,j,j->i', self.V_DOF_int.get('u').get('u'), np.ravel(self.grid.w_int_2D.get('u'), order='F'), np.ravel(self.gt.get('J').get('e').get('u'), order='F'), np.ravel(f_int[1].get('u'), order='F'))
            return np.concatenate((res_x, res_y), axis=0)
        
    def compute_continuity_volume_integral(self):
        ux = np.einsum('ji,j->ij', self.Vr_DOF_int.get('u').get('p'), np.ravel(self.gt.get('rx').get('e').get('p'), order='F')) +\
             np.einsum('ji,j->ij', self.Vs_DOF_int.get('u').get('p'), np.ravel(self.gt.get('sx').get('e').get('p'), order='F'))
        vy = np.einsum('ji,j->ij', self.Vr_DOF_int.get('u').get('p'), np.ravel(self.gt.get('ry').get('e').get('p'), order='F')) +\
             np.einsum('ji,j->ij', self.Vs_DOF_int.get('u').get('p'), np.ravel(self.gt.get('sy').get('e').get('p'), order='F'))
        ux_int = np.einsum('ij,j,j->ij', ux, np.ravel(self.gt.get('J').get('e').get('p'), order='F'), np.ravel(self.grid.w_int_2D.get('p'), order='F'))
        vy_int = np.einsum('ij,j,j->ij', vy, np.ravel(self.gt.get('J').get('e').get('p'), order='F'), np.ravel(self.grid.w_int_2D.get('p'), order='F'))
        
        res_u = -np.einsum('ij,jk->ki', ux_int, self.V_DOF_int.get('p').get('p'))
        res_v = -np.einsum('ij,jk->ki', vy_int, self.V_DOF_int.get('p').get('p'))
        return np.concatenate((res_u, res_v), axis=1)
    
    def compute_momentum_laplace_volume_integral(self, problem):
        ux = np.einsum('ji,j->ij', self.Vr_DOF_int.get('u').get('u'), np.ravel(self.gt.get('rx').get('e').get('u'), order='F')) +\
             np.einsum('ji,j->ij', self.Vs_DOF_int.get('u').get('u'), np.ravel(self.gt.get('sx').get('e').get('u'), order='F'))
        uy = np.einsum('ji,j->ij', self.Vr_DOF_int.get('u').get('u'), np.ravel(self.gt.get('ry').get('e').get('u'), order='F')) +\
             np.einsum('ji,j->ij', self.Vs_DOF_int.get('u').get('u'), np.ravel(self.gt.get('sy').get('e').get('u'), order='F'))
        
        psix = np.einsum('ji,j->ji', self.Vr_DOF_int.get('u').get('u'), np.ravel(self.gt.get('rx').get('e').get('u'), order='F')) +\
               np.einsum('ji,j->ji', self.Vs_DOF_int.get('u').get('u'), np.ravel(self.gt.get('sx').get('e').get('u'), order='F'))
        psiy = np.einsum('ji,j->ji', self.Vr_DOF_int.get('u').get('u'), np.ravel(self.gt.get('ry').get('e').get('u'), order='F')) +\
               np.einsum('ji,j->ji', self.Vs_DOF_int.get('u').get('u'), np.ravel(self.gt.get('sy').get('e').get('u'), order='F'))
        
        ux_int = np.einsum('ij,j,j->ij', ux, np.ravel(self.gt.get('J').get('e').get('u'), order='F'), np.ravel(self.grid.w_int_2D.get('u'), order='F'))
        uy_int = np.einsum('ij,j,j->ij', uy, np.ravel(self.gt.get('J').get('e').get('u'), order='F'), np.ravel(self.grid.w_int_2D.get('u'), order='F'))

        res_x = self.settings.problem.kinematic_viscosity*(np.einsum('ij,jk->ki', ux_int, psix) + np.einsum('ij,jk->ki', uy_int, psiy))
        if problem=='Poisson':
            return res_x
        elif problem=='Stokes':
            return np.concatenate((np.concatenate((res_x, np.zeros_like(res_x)), axis=1), np.concatenate((np.zeros_like(res_x), res_x), axis=1)), axis=0)

    def compute_momentum_pressure_volume_integral(self):
        psix = np.einsum('ji,j->ji', self.Vr_DOF_int.get('u').get('u'), np.ravel(self.gt.get('rx').get('e').get('u'), order='F')) +\
               np.einsum('ji,j->ji', self.Vs_DOF_int.get('u').get('u'), np.ravel(self.gt.get('sx').get('e').get('u'), order='F'))
        psiy = np.einsum('ji,j->ji', self.Vr_DOF_int.get('u').get('u'), np.ravel(self.gt.get('ry').get('e').get('u'), order='F')) +\
               np.einsum('ji,j->ji', self.Vs_DOF_int.get('u').get('u'), np.ravel(self.gt.get('sy').get('e').get('u'), order='F'))
        
        p_int = np.einsum('ji,j,j->ij', self.V_DOF_int.get('p').get('u'), np.ravel(self.gt.get('J').get('e').get('u'), order='F'), np.ravel(self.grid.w_int_2D.get('u'), order='F'))
        
        res_x = -np.einsum('ij,jk->ki', p_int, psix)
        res_y = -np.einsum('ij,jk->ki', p_int, psiy)
        return np.concatenate((res_x, res_y), axis=0)
    
    def compute_momentum_velocity_penalty_volume_integral(self):
        ux = np.einsum('ji,j->ij', self.Vr_DOF_int.get('u').get('u'), np.ravel(self.gt.get('rx').get('e').get('u'), order='F')) +\
             np.einsum('ji,j->ij', self.Vs_DOF_int.get('u').get('u'), np.ravel(self.gt.get('sx').get('e').get('u'), order='F'))
        vy = np.einsum('ji,j->ij', self.Vr_DOF_int.get('u').get('u'), np.ravel(self.gt.get('ry').get('e').get('u'), order='F')) +\
             np.einsum('ji,j->ij', self.Vs_DOF_int.get('u').get('u'), np.ravel(self.gt.get('sy').get('e').get('u'), order='F'))

        ux_int = np.einsum('ij,j,j->ij', ux, np.ravel(self.gt.get('J').get('e').get('u'), order='F'), np.ravel(self.grid.w_int_2D.get('u'), order='F')) 
        vy_int = np.einsum('ij,j,j->ij', vy, np.ravel(self.gt.get('J').get('e').get('u'), order='F'), np.ravel(self.grid.w_int_2D.get('u'), order='F')) 

        psix = np.einsum('ji,j->ji', self.Vr_DOF_int.get('u').get('u'), np.ravel(self.gt.get('rx').get('e').get('u'), order='F')) +\
               np.einsum('ji,j->ji', self.Vs_DOF_int.get('u').get('u'), np.ravel(self.gt.get('sx').get('e').get('u'), order='F'))
        psiy = np.einsum('ji,j->ji', self.Vr_DOF_int.get('u').get('u'), np.ravel(self.gt.get('ry').get('e').get('u'), order='F')) +\
               np.einsum('ji,j->ji', self.Vs_DOF_int.get('u').get('u'), np.ravel(self.gt.get('sy').get('e').get('u'), order='F'))

        res_u_x = self.grid.gamma*(np.einsum('ij,jk->ki', ux_int, psix))
        res_v_x = self.grid.gamma*(np.einsum('ij,jk->ki', vy_int, psix))
        res_u_y = self.grid.gamma*(np.einsum('ij,jk->ki', ux_int, psiy))
        res_v_y = self.grid.gamma*(np.einsum('ij,jk->ki', vy_int, psiy))
        return np.concatenate((np.concatenate((res_u_x, res_v_x), axis=1), np.concatenate((res_u_y, res_v_y), axis=1)), axis=0)

    
class CoarseElement(Element):
    def __init__(self, x_coarse, y_coarse, grid, fine_elements, coarsening_factor):
        self.coarsening_factor = coarsening_factor
        self.delta_R = 2/self.coarsening_factor
        self.grid = grid
        xy_int_coarse, gt_coarse = self._init_coarse_element(fine_elements)
        super().__init__(x=x_coarse, y=y_coarse, grid=grid, xy_int=xy_int_coarse, geometric_terms=gt_coarse)

    def _init_coarse_element(self, fine_elements):
        xy_int = {'xy_int': None,
                  'xy_int_imin': None,
                  'xy_int_imax': None,
                  'xy_int_jmin': None,
                  'xy_int_jmax': None,
        }
        geometric_terms =  {'J': {'e': {}, 'imin': {}, 'imax': {}, 'jmin': {}, 'jmax': {}}, 
                            'rx': {'e': {}, 'imin': {}, 'imax': {}, 'jmin': {}, 'jmax': {}}, 
                            'sx': {'e': {}, 'imin': {}, 'imax': {}, 'jmin': {}, 'jmax': {}}, 
                            'ry': {'e': {}, 'imin': {}, 'imax': {}, 'jmin': {}, 'jmax': {}}, 
                            'sy': {'e': {}, 'imin': {}, 'imax': {}, 'jmin': {}, 'jmax': {}}, 
                            'n': {'imin': {}, 'imax': {}, 'jmin': {}, 'jmax': {}}, 
        }

        for key in self.grid.r_int.keys():
            R_int, S_int = self.grid.r_int.get(key), self.grid.r_int.get(key)
            M, N = fine_elements.shape[0], fine_elements.shape[1]

            x_int, y_int = np.zeros((len(R_int),len(R_int))), np.zeros((len(R_int),len(R_int)))
            x_int_imin, y_int_imin = np.zeros(len(R_int)), np.zeros(len(R_int))
            x_int_imax, y_int_imax = np.zeros(len(R_int)), np.zeros(len(R_int))
            x_int_jmin, y_int_jmin = np.zeros(len(R_int)), np.zeros(len(R_int))
            x_int_jmax, y_int_jmax = np.zeros(len(R_int)), np.zeros(len(R_int))

            J_e, rx_e, sx_e, ry_e, sy_e = np.zeros((len(R_int),len(R_int))), np.zeros((len(R_int),len(R_int))), np.zeros((len(R_int),len(R_int))), np.zeros((len(R_int),len(R_int))), np.zeros((len(R_int),len(R_int)))
            J_imin, rx_imin, sx_imin, ry_imin, sy_imin, n_imin = np.zeros(len(R_int)), np.zeros(len(R_int)), np.zeros(len(R_int)), np.zeros(len(R_int)), np.zeros(len(R_int)), np.zeros((len(R_int),2))
            J_imax, rx_imax, sx_imax, ry_imax, sy_imax, n_imax = np.copy(J_imin), np.copy(rx_imin), np.copy(sx_imin), np.copy(ry_imin), np.copy(sy_imin), np.copy(n_imin)
            J_jmin, rx_jmin, sx_jmin, ry_jmin, sy_jmin, n_jmin = np.copy(J_imin), np.copy(rx_imin), np.copy(sx_imin), np.copy(ry_imin), np.copy(sy_imin), np.copy(n_imin)
            J_jmax, rx_jmax, sx_jmax, ry_jmax, sy_jmax, n_jmax = np.copy(J_imin), np.copy(rx_imin), np.copy(sx_imin), np.copy(ry_imin), np.copy(sy_imin), np.copy(n_imin)
            
            k=0
            for iS, S in enumerate(S_int):
                for iR, R in enumerate(R_int):
                    k+=1
                    l=0
                    for n in range(N):
                        if l: break
                        for m in range(M):
                            if l: break
                            r = (2*R + 2 - self.delta_R*(1+m*2))/self.delta_R
                            s = (2*S + 2 - self.delta_R*(1+n*2))/self.delta_R
                            if -1 <= r <= 1 and -1 <= s <= 1:
                                if l: break
                                l+=1
                                r, s = {key: [r]}, {key: [s]}                         

                                element = fine_elements[m,n]
                                x, y = element.x, element.y

                                x_int[iR,iS], y_int[iR,iS] = element.metric_xy_rs(x, y, r, s, vars=key, no_dict=True)
                                J_e[iR,iS], rx_e[iR,iS], sx_e[iR,iS], ry_e[iR,iS], sy_e[iR,iS] = element.compute_geometric_terms(x, y, r_int=r, s_int=s, coarsening_factor=self.coarsening_factor, vars=key, no_dict=True)

                                if R==R_int[0]:
                                    ### element has a imin boundary
                                    x_int_imin[iS], y_int_imin[iS] = element.metric_xy_rs(x, y, [-1], s, vars=key, no_dict=True)
                                    J_imin[iS], rx_imin[iS], sx_imin[iS], ry_imin[iS], sy_imin[iS], n_imin[iS,:] = element.compute_geometric_terms(x, y, r_int=[-1], s_int=s, face='imin', coarsening_factor=self.coarsening_factor, vars=key, no_dict=True)
                                elif R==R_int[-1]:
                                    ### element has a imax boundary
                                    x_int_imax[iS], y_int_imax[iS] = element.metric_xy_rs(x, y, [1], s, vars=key, no_dict=True)
                                    J_imax[iS], rx_imax[iS], sx_imax[iS], ry_imax[iS], sy_imax[iS], n_imax[iS,:] = element.compute_geometric_terms(x, y, r_int=[1], s_int=s, face='imax', coarsening_factor=self.coarsening_factor, vars=key, no_dict=True)
                                if S==S_int[0]:
                                    ### element has a jmin boundary
                                    x_int_jmin[iR], y_int_jmin[iR] = element.metric_xy_rs(x, y, r, [-1], vars=key, no_dict=True)
                                    J_jmin[iR], rx_jmin[iR], sx_jmin[iR], ry_jmin[iR], sy_jmin[iR], n_jmin[iR,:] = element.compute_geometric_terms(x, y, r_int=r, s_int=[-1], face='jmin', coarsening_factor=self.coarsening_factor, vars=key, no_dict=True)
                                elif S==S_int[-1]:
                                    ### element has a jmax boundary
                                    x_int_jmax[iR], y_int_jmax[iR] = element.metric_xy_rs(x, y, r, [1], vars=key, no_dict=True)
                                    J_jmax[iR], rx_jmax[iR], sx_jmax[iR], ry_jmax[iR], sy_jmax[iR], n_jmax[iR,:] = element.compute_geometric_terms(x, y, r_int=r, s_int=[1], face='jmax', coarsening_factor=self.coarsening_factor, vars=key, no_dict=True)

                                break
            
            ### this must be changed if the multigrid algorithm were to be extended for stokes, now only accounts for u (Poisson)
            xy_int['xy_int'] = ({key: x_int}, {key: y_int})
            xy_int['xy_int_imin'] = ({key: x_int_imin}, {key: y_int_imin})
            xy_int['xy_int_imax'] = ({key: x_int_imax}, {key: y_int_imax})
            xy_int['xy_int_jmin'] = ({key: x_int_jmin}, {key: y_int_jmin})
            xy_int['xy_int_jmax'] = ({key: x_int_jmax}, {key: y_int_jmax})

            geometric_terms['J']['e'][key] = J_e
            geometric_terms['J']['imin'][key] = J_imin
            geometric_terms['J']['imax'][key] = J_imax
            geometric_terms['J']['jmin'][key] = J_jmin
            geometric_terms['J']['jmax'][key] = J_jmax

            geometric_terms['rx']['e'][key] = rx_e
            geometric_terms['rx']['imin'][key] = rx_imin
            geometric_terms['rx']['imax'][key] = rx_imax
            geometric_terms['rx']['jmin'][key] = rx_jmin
            geometric_terms['rx']['jmax'][key] = rx_jmax

            geometric_terms['sx']['e'][key] = sx_e
            geometric_terms['sx']['imin'][key] = sx_imin
            geometric_terms['sx']['imax'][key] = sx_imax
            geometric_terms['sx']['jmin'][key] = sx_jmin
            geometric_terms['sx']['jmax'][key] = sx_jmax

            geometric_terms['ry']['e'][key] = ry_e
            geometric_terms['ry']['imin'][key] = ry_imin
            geometric_terms['ry']['imax'][key] = ry_imax
            geometric_terms['ry']['jmin'][key] = ry_jmin
            geometric_terms['ry']['jmax'][key] = ry_jmax

            geometric_terms['sy']['e'][key] = sy_e
            geometric_terms['sy']['imin'][key] = sy_imin
            geometric_terms['sy']['imax'][key] = sy_imax
            geometric_terms['sy']['jmin'][key] = sy_jmin
            geometric_terms['sy']['jmax'][key] = sy_jmax

            geometric_terms['n']['imin'][key] = n_imin
            geometric_terms['n']['imax'][key] = n_imax
            geometric_terms['n']['jmin'][key] = n_jmin
            geometric_terms['n']['jmax'][key] = n_jmax

        return xy_int, geometric_terms
                    