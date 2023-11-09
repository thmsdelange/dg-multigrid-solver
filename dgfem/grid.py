from scipy.io import FortranFile
import numpy as np
from copy import deepcopy
import os
import pickle

from dgfem.face import Face
from dgfem.element import Element, CoarseElement
from dgfem.boundary import Boundary
from dgfem.interpolation import Interpolation
from utils.helpers import compute_m, obj_to_dict
from utils.logger import Logger

class Geometry:
    def __init__(self, filepath, settings):
        self.settings = settings
        self.logger = Logger(__name__, self.settings).logger
        self.filepath = filepath
        self.P_grid = self.settings.grid.polynomial_degree
        self.N_grid = self.P_grid+1
        self.N_DOF_grid = self.N_grid**2
        self.O_grid = self.settings.grid.O_grid
        self.fully_periodic_boundaries = self.settings.grid.fully_periodic_boundaries
        self.read()

    def read(self):
        self.logger.debug(f'Reading grid from {self.filepath}')
        if 'circle' in self.filepath.lower() and not self.O_grid: self.logger.warning("It seems that you are reading a circular grid without the O-grid condition")
        with FortranFile(self.filepath, 'r', '<u4') as ff: # little endian, unassigned 4-byte integer
            ### read header and number of blocks
            header = ff.read_ints('<i4')
            nblocks = header[0]

            if nblocks.itemsize != 4: raise ValueError(f'Size of the record nblocks is {nblocks.itemsize} instead of 4')
            if nblocks != 1: raise ValueError(f'Number of blocks is {nblocks} instead of 1')

            ### read dimensions
            il, jl, kl = ff.read_ints('<i4')
            N = il*jl*kl
            dimsize = np.sum([il.itemsize, jl.itemsize, kl.itemsize])

            if dimsize != 12: raise ValueError(f'Size of the record dims is {dimsize} instead of 12')
            if kl != 1: raise ValueError(f'More than one point in third dimension')
            self.logger.debug(f'Found {il} points in x-direction and {jl} points in y-direction')
            self.logger.debug(f'Total number of grid points in the domain is {N}')

            ### read coordinates and transpose dimensions (array dimensions in Fortran are different
            ### see: https://github.com/TimoLin/pyPlot3d)
            coords = ff.read_ints('<d')
            self.x = coords[:il*jl].reshape((jl, il))
            self.y = coords[il*jl:2*il*jl].reshape((jl, il))

            self.x = np.einsum('ji->ij', self.x)
            self.y = np.einsum('ji->ij', self.y)
            
            if self.O_grid:
                if not np.all(abs(self.x[0,:] - self.x[-1,:]) < 1e-15) or not np.all(abs(self.y[0,:] - self.y[-1,:]) < 1e-15): raise ValueError('O-grid is not closed')

            ### recover the numer of elements in each direction from the LGL polynomial degree
            self.Ni = (il-1)//self.P_grid
            self.Nj = (jl-1)//self.P_grid
            self.N = self.Ni*self.Nj
            self.logger.debug(f"Total number of elements in the domain: {self.Ni}x{self.Nj}")
    
class Grid:
    def __init__(self, geometry, vars, discretization='dg'):
        for k, v in geometry.__dict__.items():  ### saving all geometry attributes to self (Grid)
            self.__dict__[k] = deepcopy(v)
        self.coarsening_factor = None
        self.vars = vars
        self.discretization = discretization
        self.BSR = None
        self.BSR_E = None
        self.BSR_D = None
        self.BSR_F = None
        self.BSR_block_A = None
        self.BSR_block_D = None
        self.BSR_block_G = None
        self.BSR_block_0 = None
        self.BSR_block_A_D = None
        self.BSR_block_Ainv = None
        self.BSR_block_Schur = None
        self.BSR_block_Schur_D = None
        self.BSR_block_DG = None
        self.BSR_block_DG_D = None
        self.RHS = None
        self.Epsilon = None
            
    def _extract_element_from_grid(self, i, j):
        idxgrid = np.ix_([*np.arange(i*self.P_grid, (i+1)*self.P_grid+1)],[*np.arange(j*self.P_grid, (j+1)*self.P_grid+1)])
        x_el = self.x[idxgrid]
        y_el = self.y[idxgrid]
        return Element(x_el, y_el, self)
    
    def initialize(self, P_sol, sigma=None, gamma=None):
        grid_path = os.path.join(os.getcwd(), 'cache', 'grid')
        grid_file = f'grid_{self.settings.problem.type}_{self.Ni}X{self.Nj}_nPoly{self.P_grid}_pSol{P_sol.get("u")}'
        if self.settings.grid.circular: grid_file += '_circle'
        if self.coarsening_factor: grid_file += f'_coarsened_{self.coarsening_factor}'
        grid_file += '.pkl'
        pickle_path = os.path.join(grid_path, grid_file)
        if not os.path.exists(pickle_path) or not self.settings.caching.enabled:
            self.P_sol = P_sol
            self.N_sol = {var: self.P_sol.get(var)+1 for var in self.vars}
            self.N_DOF_sol = {var: self.N_sol.get(var)**2 for var in self.vars}
            self.N_DOF_sol_tot = self.N_DOF_sol.get('u') if self.vars==['u'] else sum(value * 2 if key == 'u' else value for key, value in self.N_DOF_sol.items())
            self.N_int = {var: getattr(getattr(self.settings.solution, var), 'integration_polynomial_degree_factor')*self.P_sol.get(var)//2 + 1 for var in self.vars}
            self.sigma = sigma
            if not self.sigma:
                self.sigma = self.settings.problem.SIP_penalty_parameter if self.settings.problem.SIP_penalty_parameter else (self.P_sol.get('u')+1)**2*self.settings.problem.SIP_penalty_parameter_multiplier
            self.gamma = gamma
            if not self.gamma:
                self.gamma = self.settings.problem.velocity_penalty_parameter

            if self.vars == ['u', 'p']:
                self.logger.info(f'Initializing grid consisting of elements with P_grid={self.P_grid}, velocity P_sol={self.P_sol.get("u")}, pressure P_sol={self.P_sol.get("p")}')
                if self.discretization=='fvm': self.logger.warning("This grid is initialized for a FVM discretization, info below is probably not representative")
                self.logger.debug(f'The polynomial degree of the geometry is {self.P_grid} (number of degrees of freedom is {self.N_DOF_grid})')
                self.logger.debug(f'The polynomial degree of the velocity solution is {self.P_sol.get("u")} (number of degrees of freedom is {self.N_DOF_sol.get("u")})')
                self.logger.debug(f'The polynomial degree of the pressure solution is {self.P_sol.get("p")} (number of degrees of freedom is {self.N_DOF_sol.get("p")})')
                self.logger.debug(f'Using {self.N_int.get("u")} velocity integration nodes in each dimension, leading to {self.N_int.get("u")**2} integration nodes')
                self.logger.debug(f'Using {self.N_int.get("p")} pressure integration nodes in each dimension, leading to {self.N_int.get("p")**2} integration nodes')
                self.logger.debug(f'The grid consists of {self.Ni} elements in i-direction and {self.Nj} elements in j-direction')
                self.logger.debug(f'The SIP penalty term sigma of this grid is {self.sigma}')
                self.logger.debug(f'The velocity penalty term gamma of this grid is {self.gamma}')
            elif self.vars == ['u']:
                self.logger.info(f'Initializing grid consisting of elements with P_grid={self.P_grid} and P_sol={self.P_sol.get("u")}')
                if self.discretization=='fvm': self.logger.warning("This grid is initialized for a FVM discretization, info below is probably not representative")
                self.logger.debug(f'The polynomial degree of the geometry is {self.P_grid} (number of degrees of freedom is {self.N_DOF_grid})')
                self.logger.debug(f'The polynomial degree of the solution is {self.P_sol.get("u")} (number of degrees of freedom is {self.N_DOF_sol.get("u")})')
                self.logger.debug(f'Using {self.N_int.get("u")} integration nodes in each dimension, leading to {self.N_int.get("u")**2} integration nodes')
                self.logger.debug(f'The grid consists of {self.Ni} elements in i-direction and {self.Nj} elements in j-direction')
                self.logger.debug(f'The SIP penalty term sigma of this grid is {self.sigma}')
                
            self.initialize_elements()
            self.initialize_faces()
            if self.settings.problem.type == 'Stokes': self.compute_MMS_Epsilon()
            if self.settings.caching.enabled:
                with open(pickle_path, 'wb') as f:
                    pickle.dump({'grid': self, 'settings': self.settings}, f)
        else:
            with open(pickle_path, 'rb') as f:
                pickle_dict = pickle.load(f)
            assert obj_to_dict(self.settings.grid) == obj_to_dict(pickle_dict.get('settings').grid)
            assert obj_to_dict(self.settings.solution) == obj_to_dict(pickle_dict.get('settings').solution)
            assert obj_to_dict(self.settings.problem) == obj_to_dict(pickle_dict.get('settings').problem)
            self = pickle_dict.get('grid')
            
        return self
    
    def initialize_faces(self):
        self.faces_i = np.zeros((self.Ni+1, self.Nj), dtype=Face)
        for i in range(self.Ni+1):
            for j in range(self.Nj):
                if self.O_grid:
                    if i==0 or i==self.Ni:
                        self.faces_i[i,j] = Face(element_L=self.elements[-1,j], element_R=self.elements[0,j], direction='i', grid=self)
                    else:
                        self.faces_i[i,j] = Face(element_L=self.elements[i-1,j], element_R=self.elements[i,j], direction='i', grid=self)
                elif not self.O_grid:
                    if i==0:
                        self.faces_i[i,j] = Face(element_L=Boundary('imin'), element_R=self.elements[0,j], direction='i', grid=self)
                    elif i==self.Ni:
                        self.faces_i[i,j] = Face(element_L=self.elements[-1,j], element_R=Boundary('imax'), direction='i', grid=self)
                    else:
                        self.faces_i[i,j] = Face(element_L=self.elements[i-1,j], element_R=self.elements[i,j], direction='i', grid=self)

        self.faces_j = np.zeros((self.Ni, self.Nj+1), dtype=Face)
        for i in range(self.Ni):
            for j in range(self.Nj+1):
                if j==0:
                    self.faces_j[i,j] = Face(element_L=Boundary('jmin'), element_R=self.elements[i,0], direction='j', grid=self)
                elif j==self.Nj:
                    self.faces_j[i,j] = Face(element_L=self.elements[i,-1], element_R=Boundary('jmax'), direction='j', grid=self)
                else:
                    self.faces_j[i,j] = Face(element_L=self.elements[i,j-1], element_R=self.elements[i,j], direction='j', grid=self)

    def initialize_interpolation(self, N_int):
        ### initialization of terms needed in all elements
        interpol = Interpolation()

        # LGL nodes on geometry and solution grids and GL nodes for integration points
        self.r_grid = interpol.jacobi.legendre_gauss_lobatto(self.N_grid)
        self.r_sol = {key: interpol.jacobi.legendre_gauss_lobatto(self.N_sol.get(key)) if self.N_sol.get(key)>1 else np.array([0.]) for key in self.N_sol}
        rw_int = {var: interpol.jacobi.gauss_legendre_quadrature(N_int.get(var)) for var in self.vars}
        self.r_int, self.w_int = {var: rw_int.get(var)[0] for var in rw_int}, {var: rw_int.get(var)[1] for var in rw_int}   # Ideally, I want this in a single line but I don't want to call interpol.jacobi.gauss_legendre_quadrature twice
        self.w_int_2D = {var: np.outer(self.w_int.get(var), self.w_int.get(var)) for var in self.w_int}

        # Vandermonde (and its derivatives) matrices for mapping of metric terms
        self.V_grid_grid = interpol.vandermonde2D(self.N_grid, self.r_grid, self.r_grid)
        self.V_grid_int = interpol.vandermonde2D(self.N_grid, self.r_int, self.r_int)
        self.Vr_grid_int, self.Vs_grid_int = interpol.grad_vandermonde2D(self.N_grid, self.r_int, self.r_int)
        self.Vr_grid_int_imin, self.Vs_grid_int_imin = interpol.grad_vandermonde2D(self.N_grid, [-1], self.r_int)
        self.Vr_grid_int_imax, self.Vs_grid_int_imax = interpol.grad_vandermonde2D(self.N_grid, [1], self.r_int)
        self.Vr_grid_int_jmin, self.Vs_grid_int_jmin = interpol.grad_vandermonde2D(self.N_grid, self.r_int, [-1])
        self.Vr_grid_int_jmax, self.Vs_grid_int_jmax = interpol.grad_vandermonde2D(self.N_grid, self.r_int, [1])

        # Vandermonde (and its derivatives) matrices for solution states in volumes
        self.V_DOF_int = interpol.vandermonde2D(self.N_sol, self.r_int, self.r_int)
        self.Vr_DOF_int, self.Vs_DOF_int = interpol.grad_vandermonde2D(self.N_sol, self.r_int, self.r_int)

        # Vandermonde (and its derivatives) matrices for solution states on i and j faces
        self.V_DOF_int_iL = interpol.vandermonde2D(self.N_sol, [1], self.r_int)
        self.V_DOF_int_iR = interpol.vandermonde2D(self.N_sol, [-1], self.r_int)
        self.Vr_DOF_int_iL, self.Vs_DOF_int_iL = interpol.grad_vandermonde2D(self.N_sol, [1], self.r_int)
        self.Vr_DOF_int_iR, self.Vs_DOF_int_iR = interpol.grad_vandermonde2D(self.N_sol, [-1], self.r_int)
        self.V_DOF_int_jL = interpol.vandermonde2D(self.N_sol, self.r_int, [1])
        self.V_DOF_int_jR = interpol.vandermonde2D(self.N_sol, self.r_int, [-1])
        self.Vr_DOF_int_jL, self.Vs_DOF_int_jL = interpol.grad_vandermonde2D(self.N_sol, self.r_int, [1])
        self.Vr_DOF_int_jR, self.Vs_DOF_int_jR = interpol.grad_vandermonde2D(self.N_sol, self.r_int, [-1])

        # Vandermonde for translation of solution modes to grid nodes
        self.V_DOF_grid = interpol.vandermonde2D(self.N_sol, self.r_grid, self.r_grid)

    def initialize_elements(self):
        self.initialize_interpolation(self.N_int)
        self.elements = np.array([[self._extract_element_from_grid(i, j) for j in range(self.Nj)] for i in range(self.Ni)])

        ### check if elements close grid
        elements_Ni0 = self.elements[0,:]
        elements_Nim1 = self.elements[-1,:]

        if self.O_grid:
            for first, last in zip(elements_Ni0, elements_Nim1):
                if not np.all(abs(first.x[0,:] - last.x[-1,:]) < 1e-15) or not np.all(abs(first.y[0,:] - last.y[-1,:]) < 1e-15): raise ValueError(f'Element does not close O-grid with neighbouring element')      

    def compute_MMS_Epsilon(self):
        if not self.settings.solution.manufactured_solution: 
            self.Epsilon = 0.
        else:
            MMS_f_continuity = np.zeros(self.N)
            MMS_u_dot_n = np.copy(MMS_f_continuity)
            A = 0.
            for j in range(self.Nj):
                for i in range(self.Ni):
                    m = compute_m(i,j,self.Ni)
                    element = self.elements[i,j]
                    face_imin = self.faces_i[i,j]
                    face_imax = self.faces_i[i+1,j]
                    face_jmin = self.faces_j[i,j]
                    face_jmax = self.faces_j[i,j+1]

                    xy_int = element.xy_int
                    u_int_imin = self.settings.problem.exact_solution_function(*xy_int['xy_int_imin'], self.settings.problem.type, 'solution')
                    u_int_imax = self.settings.problem.exact_solution_function(*xy_int['xy_int_imax'], self.settings.problem.type, 'solution')
                    u_int_jmin = self.settings.problem.exact_solution_function(*xy_int['xy_int_jmin'], self.settings.problem.type, 'solution')
                    u_int_jmax = self.settings.problem.exact_solution_function(*xy_int['xy_int_jmax'], self.settings.problem.type, 'solution')
                    f_int_continuity = self.settings.problem.exact_solution_function(*xy_int.get('xy_int'), self.settings.problem.type, 'source_continuity')
                    A += element.A

                    if not self.O_grid:
                        if i==0: # Dirichlet boundary
                            MMS_u_dot_n[m] += face_imin.compute_continuity_MMS_u_dot_n_surface_integral(u_int_imin)
                        if i==self.Ni-1: # Dirichlet boundary
                            MMS_u_dot_n[m] += face_imax.compute_continuity_MMS_u_dot_n_surface_integral(u_int_imax)
                    if j==0: # Dirichlet boundary
                        MMS_u_dot_n[m] += face_jmin.compute_continuity_MMS_u_dot_n_surface_integral(u_int_jmin)
                    if j==self.Nj-1: # Dirichlet boundary
                        MMS_u_dot_n[m] += face_jmax.compute_continuity_MMS_u_dot_n_surface_integral(u_int_jmax)
            
                    MMS_f_continuity[m] = element.compute_continuity_MMS_volume_integral(f_int_continuity)
                    
            MMS_f_continuity_integral = np.einsum('j->', MMS_f_continuity)
            MMS_u_dot_n_integral = np.einsum('j->', MMS_u_dot_n)
            self.Epsilon = (MMS_f_continuity_integral - MMS_u_dot_n_integral)/A

            self.logger.debug(f'{MMS_f_continuity_integral=:.2e}')
            self.logger.debug(f'{MMS_u_dot_n_integral=:.2e}')
            self.logger.debug(f'{self.Epsilon=:.2e}')
        

class CoarseGrid(Grid):
    def __init__(self, geometry, fine_grid, vars, discretization='dg'):
        super().__init__(geometry, vars)
        for k, v in fine_grid.__dict__.items():     ### saving all fine_grid attributes to self (CoarseGrid)
            self.__dict__[k] = deepcopy(v)
        self.discretization = discretization
        if self.discretization == 'fvm':
            self.P_sol = {key: 0 for key in self.P_sol.keys()}


    def _extract_coarsened_elements(self, i, j, fine_elements, grid, coarsening_factor):
        idxgrid = np.ix_([*np.arange(i*self.P_grid*coarsening_factor, (i+1)*self.P_grid*coarsening_factor+1, coarsening_factor)],[*np.arange(j*self.P_grid*coarsening_factor, (j+1)*self.P_grid*coarsening_factor+1, coarsening_factor)])
        x_el_coarse = self.x[idxgrid]
        y_el_coarse = self.y[idxgrid]
        return CoarseElement(x_el_coarse, y_el_coarse, grid, fine_elements, coarsening_factor)
    
    def initialize(self, coarsening_factor):
        grid_path = os.path.join(os.getcwd(), 'cache', 'grid')
        grid_file = f'grid_{self.settings.problem.type}_{self.Ni}X{self.Nj}_nPoly{self.P_grid}_pSol{self.P_sol.get("u")}_coarsened_{coarsening_factor}'
        if self.settings.grid.circular: grid_file += '_circle'
        grid_file += '.pkl'
        pickle_path = os.path.join(grid_path, grid_file)
        if not os.path.exists(pickle_path) or not self.settings.caching.enabled:
            self.coarsening_factor = coarsening_factor
            elements_shape = self.elements.shape
            self.Ni_fine = self.Ni
            self.Nj_fine = self.Nj
            self.Ni = elements_shape[0]//self.coarsening_factor # coarse_Ni
            self.Nj = elements_shape[1]//self.coarsening_factor # coarse_Nj
            self.N = self.Ni*self.Nj
            
            if self.vars == ['u', 'p']:
                self.logger.info(f'Initializing coarsened grid (by a factor {self.coarsening_factor}) consisting of elements with P_grid={self.P_grid}, velocity P_sol={self.P_sol.get("u")}, pressure P_sol={self.P_sol.get("p")}')
                self.logger.debug(f'The polynomial degree of the geometry is {self.P_grid} (number of degrees of freedom is {self.N_DOF_grid})')
                self.logger.debug(f'The polynomial degree of the velocity solution is {self.P_sol.get("u")} (number of degrees of freedom is {self.N_DOF_sol.get("u")})')
                self.logger.debug(f'The polynomial degree of the pressure solution is {self.P_sol.get("p")} (number of degrees of freedom is {self.N_DOF_sol.get("p")})')
                self.logger.debug(f'Using {self.N_int.get("u")} velocity integration nodes in each dimension, leading to {self.N_int.get("u")**2} integration nodes')
                self.logger.debug(f'Using {self.N_int.get("p")} pressure integration nodes in each dimension, leading to {self.N_int.get("p")**2} integration nodes')
                self.logger.debug(f'The grid consists of {self.Ni} elements in i-direction and {self.Nj} elements in j-direction')
                self.logger.debug(f'The SIP penalty term sigma of this grid is {self.sigma}')
                self.logger.debug(f'The velocity penalty term gamma of this grid is {self.gamma}')
            elif self.vars == ['u']:
                self.logger.info(f'Initializing coarsened grid (by a factor {self.coarsening_factor}) consisting of elements with P_grid={self.P_grid} and P_sol={self.P_sol.get("u")}')
                self.logger.debug(f'The polynomial degree of the geometry is {self.P_grid} (number of degrees of freedom is {self.N_DOF_grid})')
                self.logger.debug(f'The polynomial degree of the solution is {self.P_sol.get("u")} (number of degrees of freedom is {self.N_DOF_sol.get("u")})')
                self.logger.debug(f'Using {self.N_int.get("u")} integration nodes in each dimension, leading to {self.N_int.get("u")**2} integration nodes')
                self.logger.debug(f'The grid consists of {self.Ni} elements in i-direction and {self.Nj} elements in j-direction')
                self.logger.debug(f'The SIP penalty term sigma of this grid is {self.sigma}')
  
            self.initialize_interpolation(self.N_int)

            

            if self.Ni == 0 or self.Nj == 0: raise ValueError(f"The number of original elements (Ni,Nj)={elements_shape} cannot be divided by a factor {self.coarsening_factor}")

            fine_elements = self.elements.reshape((self.Ni, self.coarsening_factor, self.Nj, self.coarsening_factor)).transpose((0, 2, 1, 3)) 
            self.elements = np.array([[self._extract_coarsened_elements(i,j, fine_elements[i,j], self, self.coarsening_factor) for j in range(fine_elements.shape[1])] for i in range(fine_elements.shape[0])])  

            elements_Ni0 = self.elements[0,:]
            elements_Nim1 = self.elements[-1,:]

            if self.O_grid:
                for first, last in zip(elements_Ni0, elements_Nim1):
                    if not np.all(abs(first.x[0,:] - last.x[-1,:]) < 1e-15) or not np.all(abs(first.y[0,:] - last.y[-1,:]) < 1e-15): raise ValueError(f'Element does not close O-grid with neighbouring element') 

            self.initialize_faces()
            if self.settings.problem.type == 'Stokes': self.compute_MMS_Epsilon()

            # elements_to_vtk('test_elements_coarse', self.elements)
            # elements_to_vtk('test_elements_coarse_00', np.array([[self.elements[0,0]]]))
            # elements_to_vtk('test_elements_coarse_10', np.array([[self.elements[1,0]]]))
            # elements_to_vtk('test_elements_coarse_01', np.array([[self.elements[0,1]]]))
            # elements_to_vtk('test_elements_coarse_11', np.array([[self.elements[1,1]]]))
            # elements_to_vtk('test_elements_fine_00', fine_elements[0,0])
            # elements_to_vtk('test_elements_fine_0000', fine_elements[0,0,0,0])
            # elements_to_vtk('test_elements_fine_0010', fine_elements[0,0,1,0])
            # elements_to_vtk('test_elements_fine_0003', fine_elements[0,0,0,3])
            if self.settings.caching.enabled:
                with open(pickle_path, 'wb') as f:
                    pickle.dump({'grid': self, 'settings': self.settings}, f)
        else:
            with open(pickle_path, 'rb') as f:
                pickle_dict = pickle.load(f)
            assert obj_to_dict(self.settings.grid) == obj_to_dict(pickle_dict.get('settings').grid)
            assert obj_to_dict(self.settings.solution) == obj_to_dict(pickle_dict.get('settings').solution)
            assert obj_to_dict(self.settings.problem) == obj_to_dict(pickle_dict.get('settings').problem)
            self = pickle_dict.get('grid')

        return self

    

    