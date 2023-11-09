import numpy as np
from scipy.special import eval_jacobi, roots_jacobi, gamma
from numpy.linalg import solve, inv
from math import factorial

from utils.helpers import convert_to_dict

class Interpolation:
    def __init__(self, alpha=0, beta=0) -> None:
        self.jacobi = self.Jacobi(alpha, beta)
    
    class Jacobi:
        def __init__(self, alpha, beta):
            self.alpha = alpha
            self.beta = beta

        def alpha_beta(self, alpha_offset, beta_offset):
            return self.alpha + alpha_offset, self.beta + beta_offset
        
        def evaluate_orthogonal_polynomial_GS(self, x:np.ndarray, alpha:int, beta:int, P:int, mass_matrix:np.ndarray):
            psi = self.evaluate_polynomial(x, alpha, beta, P)
            for j in range(P):
                print(f'{P=}')
                print(f'{j=}')
                print()
                psi -= mass_matrix[P,j]/mass_matrix[j,j]*self.evaluate_polynomial(x, alpha, beta, j)
            return psi

        def evaluate_polynomial(self, x:np.ndarray, alpha:int, beta:int, P:int):
            """Evaluate the P'th order jacobi polynomial at points x. The polynomials are normalized to be orthonormal
            (https://homepage.tudelft.nl/11r49/documents/wi4006/orthopoly.pdf)

            Args:
                x (np.ndarray): points at which the polynomial should be evaluated
                alpha (int): alpha
                beta (int): beta
                P (int): polynomial order

            Returns:
                tuple(ndarray(P+1)): tuple of numpy arrays containing the jacobi polynomial roots (nodes) and weights
            """   
            norm = 2**(alpha+beta+1)*gamma(P+alpha+1)*gamma(P+beta+1)/((2*P+alpha+beta+1)*gamma(P+alpha+beta+1)*factorial(P))
            jacobi = eval_jacobi(P, alpha, beta, x)
            return jacobi/np.sqrt(norm)

        def evaluate_legendre(self, x:np.ndarray, P:int, mass_matrix=None):
            if not isinstance(mass_matrix, np.ndarray):
                return self.evaluate_polynomial(x, 0, 0, P)
            else:
                return self.evaluate_orthogonal_polynomial_GS(x, 0, 0, P, mass_matrix)
            
        def evaluate_grad_legendre(self, x:np.ndarray, P:int, mass_matrix:np.ndarray):
            # if P < 1: raise ValueError('The polynomial order P must be a positive integer')
            if P == 0:
                return np.zeros_like(x)
            if not isinstance(mass_matrix, np.ndarray):
                return np.sqrt(P*(P+self.alpha+self.beta+1))*self.evaluate_polynomial(x, self.alpha+1, self.beta+1, P-1)
            else:
                return np.sqrt(P*(P+self.alpha+self.beta+1))*self.evaluate_orthogonal_polynomial_GS(x, self.alpha+1, self.beta+1, P-1, mass_matrix)

        def roots_polynomial(self, P:int, alpha_offset=0, beta_offset=0):
            """Calculate the P'th order jacobi polynomial roots

            Args:
                alpha (int): alpha
                beta (int): beta
                P (int): polynomial order

            Returns:
                tuple(ndarray(P+1)): tuple of numpy arrays containing the jacobi polynomial roots (nodes) and weights
            """        
            alpha, beta = self.alpha_beta(alpha_offset, beta_offset)
            return roots_jacobi(P, alpha, beta)

        def gauss_legendre_quadrature(self, N:int):
            """Calculate P'th order Gauss-Legendre quadrature nodes and weights

            Args:
                alpha (int): alpha
                beta (beta): beta
                N (int): number of degrees of freedom

            Returns:
                tuple(ndarray(P+1)): tuple of numpy arrays containing the GL nodes and weights
            """        
            return roots_jacobi(N, 0, 0)

        def legendre_gauss_lobatto(self, N:int):
            """Calculate P'th order Legendre-Gauss-Lobatto quadrature nodes (weights not included)

            Args:
                alpha (int): alpha
                beta (int): beta
                N (int): number of degrees of freedom

            Raises:
                ValueError: polynomial order must be > 0

            Returns:
                ndarray(P+1): numpy array containing the LGL nodes
            """        
            P = N-1
            if P < 1: raise ValueError('The polynomial order P must be a positive integer')

            xi = np.zeros(P+1)
            xi[0] ,xi[-1] = -1, 1
            
            if P > 1:
                xi[1:-1], _ = roots_jacobi(P-1, 1, 1)
            return xi

    def vandermonde1D(self, N, r, mass_matrix=None):
        V = np.zeros((len(r), N))
        for j in range(N):      
            V[:,j] = self.jacobi.evaluate_legendre(r, j, mass_matrix)
        return V
    
    def vandermonde2D(self, N, r, s, mass_matrix=None):
        """Input can be in the form of dicts. However, the vars of all dicts should be the same and should only be set in dgfem.py (not as user input)
        First dict key represents to which variable the polynomial order (nDOF) corresponds, second represents to which variable the nodes correspond
        For example:
        V_dict['u']['u'] could represent the velocity DOFs in the velocity integration nodes
        """
        N_dict, r_dict, s_dict = convert_to_dict(N, r, s)

        V_dict = {}
        for N_key in N_dict.keys():
            V_dict[N_key] = {}
            for r_key, s_key in zip(r_dict.keys(),s_dict.keys()):
                V = np.zeros((len(r_dict[r_key])*len(s_dict[s_key]), N_dict[N_key]**2))
                n = 0
                
                for j in range(N_dict[N_key]):      
                    for i in range(N_dict[N_key]):
                        ### calculate Gram-Schmid orthogonalization of legendre polynomials and save to some array,
                        ### use this array to compute volume integral which is needed for normalization, normalize and build V
                        # print(np.outer(self.jacobi.evaluate_legendre(r, i), self.jacobi.evaluate_legendre(s, j)))
                        # exit()
                        V[:,n] = np.ravel(np.outer(self.jacobi.evaluate_legendre(r_dict[r_key], i, mass_matrix), self.jacobi.evaluate_legendre(s_dict[s_key], j, mass_matrix)), order='F')
                        n += 1
                V_dict[N_key][r_key] = V
        return V_dict
    
    def grad_vandermonde1D(self, N, r):
        Vr = np.zeros((len(r), N))
        for j in range(1, N):      
            Vr[:,j] = self.jacobi.evaluate_grad_legendre(r, j)
        return Vr
    
    def grad_vandermonde2D(self, N, r, s, mass_matrix=None):
        """Input can be in the form of dicts. However, the vars of all dicts should be the same and should only be set in dgfem.py (not as user input)
        First dict key represents to which variable the polynomial order (nDOF) corresponds, second represents to which variable the nodes correspond
        For example:
        V_dict['u']['u'] could represent the velocity DOFs in the velocity integration nodes
        """
        N_dict, r_dict, s_dict = convert_to_dict(N, r, s)

        Vr_dict, Vs_dict = {}, {}
        for N_key in N_dict.keys():
            Vr_dict[N_key], Vs_dict[N_key] = {}, {}
            for r_key, s_key in zip(r_dict.keys(),s_dict.keys()):
                Vr, Vs = np.zeros((len(r_dict[r_key])*len(s_dict[s_key]), N_dict[N_key]**2)), np.zeros((len(r_dict[r_key])*len(s_dict[s_key]), N_dict[N_key]**2))
                n = 0
                for j in range(N_dict[N_key]):      
                    for i in range(N_dict[N_key]):
                        Vr[:,n] = np.ravel(np.outer(self.jacobi.evaluate_grad_legendre(r_dict[r_key], i, mass_matrix), self.jacobi.evaluate_legendre(s_dict[s_key], j, mass_matrix)), order='F')
                        Vs[:,n] = np.ravel(np.outer(self.jacobi.evaluate_legendre(r_dict[r_key], i, mass_matrix), self.jacobi.evaluate_grad_legendre(s_dict[s_key], j, mass_matrix)), order='F')
                        n += 1
                Vr_dict[N_key][r_key], Vs_dict[N_key][r_key] = Vr, Vs
        return Vr_dict, Vs_dict

    def lagrange_basis(self, x, xi, N):
        lag_basis = np.zeros(N)
        for i in range(N):
            lj = 1.
            for j in range(len(xi)):
                if j==i:
                    continue
                lj *= (x - xi[j])/(xi[i] - xi[j])
            lag_basis[i] = lj
        return lag_basis

    def legendre_to_lagrange1D(self, r):
        Vg = self.vandermonde1D(self.Pg, self.r_LGL)
        Leg = np.array([self.jacobi.evaluate_legendre(r, p) for p in range(self.Pg+1)])
        VgTinv = inv(Vg.T)
        return np.einsum('ij,jk->ki', VgTinv, Leg)   # first row corresponds to r0, second to r1, etc. columns correspond to P0, P1,, etc
    
    def legendre_to_lagrange2D(self, r):
        Vg = self.vandermonde2D(self.Pg, self.r_LGL, self.r_LGL)
        P = self.Pg
        Lag = np.zeros(((P+1)**2, len(r)))
        m = 0
        for i in range(P+1):
            for j in range(P+1):
                Lag[m,:] = self.jacobi.evaluate_legendre(r, i) * self.jacobi.evaluate_legendre(r, j)
                m += 1

        VgTinv = inv(Vg.T)
        return np.einsum('ij,jk->ki', VgTinv, Lag)   # first row corresponds to r0, second to r1, etc. columns correspond to P0, P1,, etc

    def orthonormalize_Gram_Schmidt(self, V_DOF_int, J, w_int):
        """Orthonormalize the basis on the physical space using the Gram-Schmidt orthonormalisation technique
            !! Note that the lower basis functions are not yet defined using the p-order !!
        """
        weights = np.zeros((V_DOF_int.shape[1],V_DOF_int.shape[1]))
        V_int_DOF_orthogonal = np.copy(V_DOF_int)
        for i in range(V_DOF_int.shape[1]):
            weights[i,i] = 1.
            for j in range(i):
                weight = -(V_int_DOF_orthogonal[:,i] * V_int_DOF_orthogonal[:,j] * np.ravel(J, order='F') @ np.ravel(w_int, order='F'))/(V_int_DOF_orthogonal[:,j] * V_int_DOF_orthogonal[:,j] * np.ravel(J, order='F') @ np.ravel(w_int, order='F') + 1.e-16)
                V_int_DOF_orthogonal[:,i] += weight*V_int_DOF_orthogonal[:,j]
                weights[j,i] += weight

        norms = np.zeros(V_int_DOF_orthogonal.shape[1])
        for j in range(V_int_DOF_orthogonal.shape[1]):
            norms[j] = 1./np.sqrt(V_int_DOF_orthogonal[:,j] * V_int_DOF_orthogonal[:,j] * np.ravel(J, order='F') @ np.ravel(w_int, order='F') + 1.e-16)
        V_int_DOF_orthonormal = V_int_DOF_orthogonal*norms
        return V_int_DOF_orthonormal, weights, norms
    
    def orthonormalize_Gram_Schmidt_Extended(self, V_DOF_int, J, w_int):
        """Orthonormalize the basis on the physical space using the Gram-Schmidt orthonormalisation technique
            !! Note that the lower basis functions are not yet defined using the p-order !!
        """
        if isinstance(V_DOF_int, dict):
            V_DOF_int_orthonormal_dict = {}
            weights_dict = {}
            norms_dict = {}
            for N_key, items in V_DOF_int.items():
                V_DOF_int_orthonormal_dict[N_key] = {}
                weights_dict[N_key] = {}
                norms_dict[N_key] = {}
                for r_key, V_DOF_int in items.items():
                    weights = np.zeros((V_DOF_int.shape[1],V_DOF_int.shape[1]))
                    V_DOF_int_orthogonal = np.copy(V_DOF_int)
                    for i in range(V_DOF_int.shape[1]):
                        weights[i,i] = 1.
                        for j in range(i):
                            weight = -(V_DOF_int_orthogonal[:,i] * V_DOF_int_orthogonal[:,j] * np.ravel(J.get(r_key), order='F') @ np.ravel(w_int.get(r_key), order='F'))/(V_DOF_int_orthogonal[:,j] * V_DOF_int_orthogonal[:,j] * np.ravel(J.get(r_key), order='F') @ np.ravel(w_int.get(r_key), order='F') + 1.e-16)
                            V_DOF_int_orthogonal[:,i] += weight*V_DOF_int_orthogonal[:,j]
                            weights[j,i] += weight


                    norms = np.zeros(V_DOF_int_orthogonal.shape[1])
                    for j in range(V_DOF_int_orthogonal.shape[1]):
                        norms[j] = 1./np.sqrt(V_DOF_int_orthogonal[:,j] * V_DOF_int_orthogonal[:,j] * np.ravel(J.get(r_key), order='F') @ np.ravel(w_int.get(r_key), order='F') + 1.e-16)
                    V_DOF_int_orthonormal = V_DOF_int_orthogonal*norms
                    V_DOF_int_orthonormal_dict[N_key][r_key] = V_DOF_int_orthonormal
                    weights_dict[N_key][r_key] = weights
                    norms_dict[N_key][r_key] = norms
            return V_DOF_int_orthonormal, weights, norms
        else:
            weights = np.zeros((V_DOF_int.shape[1],V_DOF_int.shape[1]))
            V_DOF_int_orthogonal = np.copy(V_DOF_int)
            for i in range(V_DOF_int.shape[1]):
                weights[i,i] = 1.
                for j in range(i):
                    weight = -(V_DOF_int_orthogonal[:,i] * V_DOF_int_orthogonal[:,j] * np.ravel(J, order='F') @ np.ravel(w_int, order='F'))/(V_DOF_int_orthogonal[:,j] * V_DOF_int_orthogonal[:,j] * np.ravel(J, order='F') @ np.ravel(w_int, order='F') + 1.e-16)
                    V_DOF_int_orthogonal[:,i] += weight*V_DOF_int_orthogonal[:,j]
                    weights[j,i] += weight


            norms = np.zeros(V_DOF_int_orthogonal.shape[1])
            for j in range(V_DOF_int_orthogonal.shape[1]):
                norms[j] = 1./np.sqrt(V_DOF_int_orthogonal[:,j] * V_DOF_int_orthogonal[:,j] * np.ravel(J, order='F') @ np.ravel(w_int, order='F') + 1.e-16)
            V_DOF_int_orthonormal = V_DOF_int_orthogonal*norms
            return V_DOF_int_orthonormal, weights, norms