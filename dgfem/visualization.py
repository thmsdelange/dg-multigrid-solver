import numpy as np
from pyevtk.hl import gridToVTK     # https://github.com/pyscience-projects/pyevtk
import os

from dgfem.interpolation import Interpolation
interp = Interpolation()

### this file should be converted to a class with settings (object) input
### the functions in this file were mostly used for my postprocessing, so probably not extremely useful for everyone
from input import params
from dgfem.settings import Settings
settings = Settings(params)
from utils.logger import Logger
logger = Logger(__name__, settings=settings).logger 

from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
import scienceplots
plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.figsize': (3.3, 3.3)})
ieee_size = (3.3, 2.5)
rc_params = {'figure.figsize': ieee_size, 'lines.markersize':3, 'lines.linewidth':.7}
markers = ['o', '^', 's', 'v', 'd', 'p']
lines = ['solid', 'dashed', 'dotted', 'dashdot', (5, (10,3)), (0, (3, 1, 1, 1))]

### create results folders if not exist
amplification_factors_results = r'./postprocessing/plots/amplification_factors'
if not os.path.exists(amplification_factors_results):
    os.makedirs(amplification_factors_results)

basis_functions_results = r'./postprocessing/plots/basis_functions'
if not os.path.exists(basis_functions_results):
    os.makedirs(basis_functions_results)

grid_convergence_results = r'./postprocessing/plots/grid_convergence'
if not os.path.exists(grid_convergence_results):
    os.makedirs(grid_convergence_results)

spectral_radius_results = r'./postprocessing/plots/spectral_radius'
if not os.path.exists(spectral_radius_results):
    os.makedirs(spectral_radius_results)

multigrid_results = r'./postprocessing/plots/multigrid'
if not os.path.exists(multigrid_results):
    os.makedirs(multigrid_results)

smoother_results = r'./postprocessing/plots/smoother'
if not os.path.exists(smoother_results):
    os.makedirs(smoother_results)

def grid_to_vtk(filename, x, y):
    results_filepath = os.path.join(os.getcwd(), filename)

    ### convert 2D grid to 3D
    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=2)
    z = np.zeros_like(x)
    
    try:
        gridToVTK(results_filepath, x, y, z) # expand with cellData={"pressure": pressure}, pointData={"temp": temp}
        logger.debug(f'Grid visualization file written to {results_filepath}')
    except Exception as e:
        logger.error(f'Error when writing grid to file: {e}')

def elements_to_vtk(filename, elements, problem, cell_data=None, point_data=None):
    ### spread elements to x and y and pass to grid_to_vtk
    ### or export to .vtu (unstructured) if possible
    ### attach i and j as cellData (if possible)
    results_filepath = os.path.join(os.getcwd(), filename)
    x_el = np.array([[elements[i,j].x for j in range(elements.shape[1])] for i in range(elements.shape[0])])
    y_el = np.array([[elements[i,j].y for j in range(elements.shape[1])] for i in range(elements.shape[0])])
    x = np.expand_dims(x_el.transpose(0, 2, 1, 3).reshape(x_el.shape[0]*x_el.shape[2], x_el.shape[1]*x_el.shape[3]), axis=2)
    y = np.expand_dims(y_el.transpose(0, 2, 1, 3).reshape(y_el.shape[0]*y_el.shape[2], y_el.shape[1]*y_el.shape[3]), axis=2)
    z = np.zeros_like(x)
    
    # if cell_data:
    #     pass

    if point_data:
        point_data_dict = {}
        if problem == 'Stokes':
            point_data_dict['velocity'] = []
            point_data_dict['velocity_exact'] = []
        for key, data in sorted(list(point_data.items()), key=lambda x:x[0].lower(), reverse=True):
            data = np.expand_dims(data.transpose(0, 2, 1, 3).reshape(data.shape[0]*data.shape[2], data.shape[1]*data.shape[3]), axis=2)
            if problem == 'Stokes':
                if key == 'u':
                    point_data_dict['velocity'].insert(0, data)
                elif key == 'v':
                    point_data_dict['velocity'].insert(1, data)
                elif key == 'u_exact':
                    point_data_dict['velocity_exact'].insert(0, data)
                elif key == 'v_exact':
                    point_data_dict['velocity_exact'].insert(1, data)
                else:
                    point_data_dict[key] = data
            else:
                point_data_dict[key] = data
        
        if problem == 'Stokes':
            if point_data_dict['velocity']:
                point_data_dict['velocity'].append(np.zeros_like(point_data_dict['velocity'][0]))
                point_data_dict['velocity'] = tuple(point_data_dict['velocity'])
            if point_data_dict['velocity_exact']:
                point_data_dict['velocity_exact'].append(np.zeros_like(point_data_dict['velocity_exact'][0]))
                point_data_dict['velocity_exact'] = tuple(point_data_dict['velocity_exact'])

    try:
        if point_data:
            gridToVTK(results_filepath, x, y, z, pointData=point_data_dict)
        else:
            gridToVTK(results_filepath, x, y, z) # expand with cellData={"pressure": pressure}, pointData={"temp": temp}
    
        logger.debug(f'Elements visualization file written to {results_filepath}')
    except Exception as e:
        logger.error(f'Error when writing elements to file: {e}')

def modal_to_vtk(filename, grid, elements, cell_data=None, u_modal=None):
    ### interpolating the solution from the modes to the grid nodes
    u_nodal = np.zeros((grid.Ni,grid.Nj,grid.N_grid,grid.N_grid))
    u_modal = u_modal.reshape(grid.Ni,grid.Nj,grid.N_DOF_sol)
    logger.debug("Interpolating the solution from modes to nodes ...")
    for j in range(grid.Nj):
        for i in range(grid.Ni):
            u_nodal[i,j,:,:] = (grid.elements[i,j].V_DOF_grid @ u_modal[j,i,:]).reshape(grid.N_grid,grid.N_grid)

    elements_to_vtk(filename, elements, cell_data, point_data={'u': u_nodal})
    
def grid_to_vtk_lowlevel(filename, x, y):
    from pyevtk.vtk import VtkFile, VtkStructuredGrid
    results_filepath = os.path.join(os.getcwd(), filename)

    ### convert 2D grid to 3D
    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=2)
    z = np.zeros_like(x)
    start = (0,0,0)
    end = tuple(np.subtract(x.shape, (0,0,0)))

    w = VtkFile(results_filepath, VtkStructuredGrid)
    w.openGrid(start, end)
    w.openPiece(start, end)
    
    # # point data
    # temp = np.random.rand(x.size)
    # w.openData('Point', scalars='Temperature')
    # w.addData('Temperature', temp)
    # w.closeData('Point')

    # # cell data
    # pressure = np.zeros_like(x)
    # w.openData('Cell', scalars='Pressure')
    # w.addData('Pressure', pressure)
    # w.closeData('Cell')

    # coordinates of cell vertices
    w.openElement('Coordinates')
    w.addData('x_coordinates', x)
    w.addData('y_coordinates', y)
    w.addData('z_coordinates', z)
    w.closeElement('Coordinates')

    w.closePiece()
    w.closeGrid()

    # w.appendData(data=temp)
    # w.appendData(data=pressure)
    w.appendData(x)
    w.appendData(y)
    w.appendData(z)
    w.save()

def plot_standard_element(r_sol, r_int, export=False):
    rr_sol, ss_sol = np.meshgrid(r_sol, r_sol)
    rr_int, ss_int = np.meshgrid(r_int, r_int)
    
    plt.figure() if export else plt.figure(dpi=150)
    plt.scatter(rr_sol, ss_sol, facecolor='none', edgecolor='black', marker='o', clip_on=False, label='Degrees of freedom')
    plt.scatter(rr_int, ss_int, color='black', marker='x', clip_on=False, label='Integration nodes')
    plt.xlabel(r'$r$')
    plt.ylabel(r'$s$')
    plt.xticks([-1, -.5, 0, .5, 1])
    plt.yticks([-1, -.5, 0, .5, 1])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', bottom=True, left=True, pad=10)
    if export:
        plt.savefig(r'./postprocessing/plots/standard_element.svg', bbox_inches='tight')
    else:
        plt.show()

def plot_sparsity_pattern(grid, precision=1e-3, markersize=None, color=None, export=False):
    plt.figure() if export else plt.figure(dpi=150)
    plt.spy(grid.BSR.toarray(), precision=precision)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    if export:
        plt.savefig(rf'./postprocessing/sparsity_pattern_{grid.Ni}X{grid.Nj}_nPoly{grid.P_grid}_Pu{grid.P_sol.get("u")}_Pp{grid.P_sol.get("p")}.svg', bbox_inches='tight')
    else:
        plt.show()

def plot_amplification_factor(A, theta_x, theta_y, grid, export=False, suffix=None):
    Theta_x, Theta_y = np.meshgrid(theta_x, theta_y)

    plt.style.use(['science'])
    plt.rcParams.update({'figure.figsize': ieee_size})
    plt.figure() if export else plt.figure(dpi=150)
    ax = plt.axes(projection="3d")
    ax.view_init(30, 210)

    surf = ax.plot_surface(Theta_x, Theta_y, A, cmap=cm.jet)
    surf.set_clim(0, 1)
    plt.colorbar(surf, shrink=0.5, aspect=5)

    tick_locations = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$']
    ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
    ax.yaxis.set_major_locator(ticker.FixedLocator(tick_locations))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(axis='x', pad=-5)  # Adjust the pad to bring the labels closer to the axes
    ax.tick_params(axis='y', pad=-5)  # Adjust the pad to bring the labels closer to the axes
    ax.set_xlabel(r'$\theta_x$', labelpad=-8)
    ax.set_ylabel(r'$\theta_y$', labelpad=-8)
    ax.set_zlim(0, 1)
    ax.set_zticklabels([])
    if export:
        plt.savefig(rf'./postprocessing/plots/amplification_factors/amplification_factor_{grid.discretization}_{grid.Ni}X{grid.Nj}_nPoly{grid.P_sol.get("u")}_{suffix}.svg', bbox_inches='tight')
    else:
        plt.show()

def plot_nodal_basis_functions(N, export=False):
    # xi = np.linspace(-1, 1, N)
    xi = interp.jacobi.legendre_gauss_lobatto(N)
    xp = np.linspace(-1, 1, 101)
    
    ell = np.zeros((len(xi),len(xp)))
    for i,xpi in enumerate(xp):
        ell[:,i] = interp.lagrange_basis(xpi, xi, N)

    plt.style.use(['science'])
    plt.rcParams.update({'figure.figsize': ieee_size})
    plt.figure() if export else plt.figure(dpi=150)
    for i in range(N):
        plt.plot(xp,ell[i,:], label=rf'$\ell_{i}$')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\ell(x)$')
    plt.xlim([-1, 1])
    # plt.ylim([-1.2, 1.2])
    plt.legend(bbox_to_anchor = (1.28, 0.5), loc='center right')
    ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    if export:
        plt.savefig(r'./postprocessing/plots/basis_functions/nodal_basis_functions.svg', bbox_inches='tight')
    else:
        plt.show()
        
def plot_modal_basis_functions(N, export=False):
    from scipy.special import eval_jacobi
    xp = np.linspace(-1, 1, 101)
    
    psi = np.zeros((N,len(xp)))
    for i in range(N):
        psi[i,:] = eval_jacobi(i, 0, 0, xp)     # using eval_jacobi with alpha=beta=0 instead of orthonormalised interpol.jacobi.evaluate_legendre

    plt.style.use(['science'])
    plt.rcParams.update({'figure.figsize': ieee_size})
    plt.figure() if export else plt.figure(dpi=150)
    for i in range(N):
        plt.plot(xp,psi[i,:], label=rf'$\psi_{i}$')
    
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\psi(x)$')
    plt.xlim([-1, 1])
    # plt.ylim([-1.2, 1.2])
    plt.legend(bbox_to_anchor = (1.28, 0.5), loc='center right')
    ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    if export:
        plt.savefig(r'./postprocessing/plots/basis_functions/modal_basis_functions.svg', bbox_inches='tight')
    else:
        plt.show()
        
def runge_function(x):
    return 1./(1.+50.*x**2)

def calc_Lebesgue_constant(N):
    xi_equi = np.linspace(-1, 1, N)
    xi_LGL = interp.jacobi.legendre_gauss_lobatto(N)
    xp = np.linspace(-1, 1, 1001)
    
    ell_equi = np.zeros((len(xi_equi),len(xp)))
    ell_LGL = np.zeros((len(xi_LGL),len(xp)))
    for i,xpi in enumerate(xp):
        ell_equi[:,i] = interp.lagrange_basis(xpi, xi_equi, N)
        ell_LGL[:,i] = interp.lagrange_basis(xpi, xi_LGL, N)
    Lambda_P_equi = np.max(np.einsum('ij->j', np.abs(ell_equi)))
    Lambda_P_LGL = np.max(np.einsum('ij->j', np.abs(ell_LGL)))
    return Lambda_P_equi, Lambda_P_LGL

def plot_nodal_basis_functions_lebesgue_runge(N, export=False):
    xi_equi = np.linspace(-1, 1, N)
    xi_LGL = interp.jacobi.legendre_gauss_lobatto(N)
    fi_equi = runge_function(xi_equi)
    fi_LGL = runge_function(xi_LGL)
    xp = np.linspace(-1, 1, 101)
    fp = runge_function(xp)
    
    ell_equi = np.zeros((len(xi_equi),len(xp)))
    ell_LGL = np.zeros((len(xi_LGL),len(xp)))
    for i,xpi in enumerate(xp):
        ell_equi[:,i] = interp.lagrange_basis(xpi, xi_equi, N)
        ell_LGL[:,i] = interp.lagrange_basis(xpi, xi_LGL, N)
    lambda_P_equi = np.einsum('ij->j', np.abs(ell_equi))
    lambda_P_LGL = np.einsum('ij->j', np.abs(ell_LGL))
    fn_equi = np.einsum('ij,i->j', ell_equi, fi_equi)
    fn_LGL = np.einsum('ij,i->j', ell_LGL, fi_LGL)

    NN = np.linspace(2,N,N-1).astype(int)
    Lambda_equi = np.zeros(len(NN))
    Lambda_LGL = np.zeros(len(NN))
    for i,nn in enumerate(NN):
        print(calc_Lebesgue_constant(nn))
        Lambda_equi[i], Lambda_LGL[i] = calc_Lebesgue_constant(nn)

    plt.style.use(['science'])
    plt.rcParams.update({'figure.figsize': ieee_size})
    plt.figure() if export else plt.figure(dpi=150)
    for i in range(N):
        plt.plot(xp,ell_equi[i,:])
    plt.plot(xp, lambda_P_equi, '--k')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\ell_i(x)$')
    plt.xlim([-1, 1])
    # plt.ylim([-4, 12])
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    if export:
        plt.savefig(r'./postprocessing/plots/basis_functions/nodal_basis_functions_lebesgue_equi.svg', bbox_inches='tight')
    else:
        plt.show()

    plt.figure() if export else plt.figure(dpi=150)
    for i in range(N):
        plt.plot(xp,ell_LGL[i,:])
    plt.plot(xp, lambda_P_LGL, '--k')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\ell_i(x)$')
    plt.xlim([-1, 1])
    # plt.ylim([-.5, 2.1])
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    if export:
        plt.savefig(r'./postprocessing/plots/basis_functions/nodal_basis_functions_lebesgue_LGL.svg', bbox_inches='tight')
    else:
        plt.show()

    plt.style.use(['science', 'ieee'])
    plt.figure() if export else plt.figure(dpi=150)
    plt.plot(xp, fp, label='Analytical')
    plt.plot(xp, fn_equi, label='Equidistant Lagrange')
    plt.plot(xp, fn_LGL, label='LGL Lagrange')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.xlim([-1, 1])
    # plt.ylim([-1.6, 1.1])
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    if export:
        plt.savefig(r'./postprocessing/plots/basis_functions/nodal_basis_functions_runge.svg', bbox_inches='tight')
    else:
        plt.show()

    plt.style.use(['science', 'ieee'])
    plt.figure() if export else plt.figure(dpi=150)
    plt.plot(NN-1, Lambda_equi, label='Equidistant')
    plt.plot(NN-1, Lambda_LGL, label='LGL')
    plt.xlabel(r'$p$')
    plt.ylabel(r'$\Lambda_p$')
    plt.xlim([1, N-1])
    # plt.ylim([-1.6, 1.1])
    plt.legend()
    ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    if export:
        plt.savefig(r'./postprocessing/plots/basis_functions/nodal_basis_functions_lebesgue_constant.svg', bbox_inches='tight')
    else:
        plt.show()

def plot_grid_convergence(errors_dict, export=False, shape=''):
    # print(errors_dict)
    if False:
        plt.style.use(['science'])
        plt.rcParams.update(rc_params)
        plt.figure() if export else plt.figure(dpi=150)
        plt.tick_params(axis='x', which='minor')
        # plt.loglog(errors_dict.get('N'), -1e-4*np.array(errors_dict.get('N'))**5, ':k')
    
        i = 0
        for d in range(len(errors_dict.get('P'))):
            label = f'u, p={errors_dict.get("P")[d]}' if isinstance(errors_dict.get('L1_error').get('v'), np.ndarray) else f'p={errors_dict.get("P")[d]}'
            plt.loglog(errors_dict.get('N'), errors_dict.get('L1_error').get('u')[:,d], '--k', label=label, marker=markers[i])
            i+=1
            
            ax = plt.gca()
            tick_locations = errors_dict.get('N')
            tick_labels =  [f"{n}X{n}" for n in tick_locations]
            ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
            ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
            ax.set_xticklabels(tick_labels)
            ax.get_xaxis().set_tick_params(which='minor', size=0)
            ax.get_xaxis().set_tick_params(which='minor', width=0) 
            if isinstance(errors_dict.get('L1_error').get('v'), np.ndarray): plt.loglog(errors_dict.get('N'), errors_dict.get('L1_error').get('v')[:,d], label=f'v, p={errors_dict.get("P")[d]}')
            if isinstance(errors_dict.get('L1_error').get('p'), np.ndarray): plt.loglog(errors_dict.get('N'), errors_dict.get('L1_error').get('p')[:,d], label=f'p, p={errors_dict.get("P")[d]}')
        plt.xlabel('grid')
        plt.ylabel(r'$L_1$')
        plt.legend()

        if export:
            plt.savefig(rf'./postprocessing/plots/grid_convergence/{filename}_L1.svg', bbox_inches='tight')
        else:
            plt.show()
    
    if not isinstance(errors_dict.get('L1_error').get('v'), np.ndarray):
        plt.style.use(['science'])
        plt.rcParams.update(rc_params)
        plt.figure() if export else plt.figure(dpi=150)
        i = 0
        for d in range(len(errors_dict.get('P'))):
            label = rf'$p={errors_dict.get("P")[d]}$'
            plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('u')[:,d], '--k', label=label, marker=markers[i])
            i+=1
            
            ax = plt.gca()
            tick_locations = errors_dict.get('N')
            tick_labels =  [f"{n}X{n}" for n in tick_locations]
            ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
            ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
            ax.set_xticklabels(tick_labels)
            # ax.set_ylim([1e-10, 1e0]) if shape=='rectangle' else ax.set_ylim([1e-4, 1e1])
            ax.get_xaxis().set_tick_params(which='minor', size=0)
            ax.get_xaxis().set_tick_params(which='minor', width=0)
            # if isinstance(errors_dict.get('L2_error').get('v'), np.ndarray): plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('v')[:,d], label=f'v, p={errors_dict.get("P")[d]}')
            # if isinstance(errors_dict.get('L2_error').get('p'), np.ndarray): plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('p')[:,d], label=f'p, p={errors_dict.get("P")[d]}')
        plt.xlabel('grid')
        plt.ylabel(r'$L_2$(u)')
        plt.legend()
        origins_y = np.exp((np.log(errors_dict.get('L2_error').get('u')[-2,:]) + np.log(errors_dict.get('L2_error').get('u')[-1,:]))/2)
        origin_x = np.exp((np.log(32)+np.log(64))/2)
        fig = plt.gcf()
        draw_loglog_slope(fig, ax, (origin_x,origins_y[0]+origins_y[0]/3), .07, -2, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[1]+origins_y[1]/3), .07, -3, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[2]+origins_y[2]/3), .07, -4, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[3]+origins_y[3]/3), .07, -5, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[4]+origins_y[4]/3), .07, -6, color='black')

        if export:
            plt.savefig(rf'./postprocessing/plots/grid_convergence/grid_convergence_Poisson_{shape}_L2.svg', bbox_inches='tight')
        else:
            plt.show()
    else:
        ### u
        plt.style.use(['science'])
        plt.rcParams.update(rc_params)
        plt.figure() if export else plt.figure(dpi=150)
        i = 0
        for d in range(len(errors_dict.get('P'))):
            label = rf'$p={errors_dict.get("P")[d]}$'
            plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('u')[:,d], '--k', label=label, marker=markers[i])
            i+=1
            
            ax = plt.gca()
            tick_locations = errors_dict.get('N')
            tick_labels =  [f"{n}X{n}" for n in tick_locations]
            ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
            ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
            ax.set_xticklabels(tick_labels)
            # ax.set_ylim([1e-10, 1e0]) if shape=='rectangle' else ax.set_ylim([1e-10, 1e1])
            ax.get_xaxis().set_tick_params(which='minor', size=0)
            ax.get_xaxis().set_tick_params(which='minor', width=0)
            # if isinstance(errors_dict.get('L2_error').get('v'), np.ndarray): plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('v')[:,d], label=f'v, p={errors_dict.get("P")[d]}')
            # if isinstance(errors_dict.get('L2_error').get('p'), np.ndarray): plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('p')[:,d], label=f'p, p={errors_dict.get("P")[d]}')
        plt.xlabel('grid')
        plt.ylabel(r'$L_2$(u)')
        plt.legend()
        origins_y = np.exp((np.log(errors_dict.get('L2_error').get('u')[-2,:]) + np.log(errors_dict.get('L2_error').get('u')[-1,:]))/2)
        origin_x = np.exp((np.log(16)+np.log(32))/2)
        fig = plt.gcf()
        draw_loglog_slope(fig, ax, (origin_x,origins_y[0]+origins_y[0]/3), -.23, -3, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[1]+origins_y[1]/3), -.23, -4, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[2]+origins_y[2]/3), -.23, -5, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[3]+origins_y[3]/3), -.23, -6, color='black')

        if export:
            plt.savefig(rf'./postprocessing/plots/grid_convergence/grid_convergence_Stokes_u_{shape}_L2.svg', bbox_inches='tight')
        else:
            plt.show()
        
        ### v
        plt.style.use(['science'])
        plt.rcParams.update(rc_params)
        plt.figure() if export else plt.figure(dpi=150)
        i = 0
        for d in range(len(errors_dict.get('P'))):
            label = rf'$p={errors_dict.get("P")[d]}$'
            plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('v')[:,d], '--k', label=label, marker=markers[i])
            i+=1
            
            ax = plt.gca()
            tick_locations = errors_dict.get('N')
            tick_labels =  [f"{n}X{n}" for n in tick_locations]
            ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
            ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
            ax.set_xticklabels(tick_labels)
            # ax.set_ylim([1e-10, 1e0]) if shape=='rectangle' else ax.set_ylim([1e-10, 1e1])
            ax.get_xaxis().set_tick_params(which='minor', size=0)
            ax.get_xaxis().set_tick_params(which='minor', width=0)
            # if isinstance(errors_dict.get('L2_error').get('v'), np.ndarray): plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('v')[:,d], label=f'v, p={errors_dict.get("P")[d]}')
            # if isinstance(errors_dict.get('L2_error').get('p'), np.ndarray): plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('p')[:,d], label=f'p, p={errors_dict.get("P")[d]}')
        plt.xlabel('grid')
        plt.ylabel(r'$L_2$(v)')
        plt.legend()
        origins_y = np.exp((np.log(errors_dict.get('L2_error').get('v')[-2,:]) + np.log(errors_dict.get('L2_error').get('v')[-1,:]))/2)
        origin_x = np.exp((np.log(16)+np.log(32))/2)
        fig = plt.gcf()
        draw_loglog_slope(fig, ax, (origin_x,origins_y[0]+origins_y[0]/3), -.23, -3, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[1]+origins_y[1]/3), -.23, -4, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[2]+origins_y[2]/3), -.23, -5, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[3]+origins_y[3]/3), -.23, -6, color='black')

        if export:
            plt.savefig(rf'./postprocessing/plots/grid_convergence/grid_convergence_Stokes_v_{shape}_L2.svg', bbox_inches='tight')
        else:
            plt.show()
        
        ### p
        plt.style.use(['science'])
        plt.rcParams.update(rc_params)
        plt.figure() if export else plt.figure(dpi=150)
        i = 0
        for d in range(len(errors_dict.get('P'))):
            label = rf'$p={errors_dict.get("P")[d]}$'
            plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('p')[:,d], '--k', label=label, marker=markers[i])
            i+=1
            
            ax = plt.gca()
            tick_locations = errors_dict.get('N')
            tick_labels =  [f"{n}X{n}" for n in tick_locations]
            ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
            ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
            ax.set_xticklabels(tick_labels)
            # ax.set_ylim([1e-10, 1e0]) if shape=='rectangle' else ax.set_ylim([1e-10, 1e1])
            ax.get_xaxis().set_tick_params(which='minor', size=0)
            ax.get_xaxis().set_tick_params(which='minor', width=0)
            # if isinstance(errors_dict.get('L2_error').get('v'), np.ndarray): plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('v')[:,d], label=f'v, p={errors_dict.get("P")[d]}')
            # if isinstance(errors_dict.get('L2_error').get('p'), np.ndarray): plt.loglog(errors_dict.get('N'), errors_dict.get('L2_error').get('p')[:,d], label=f'p, p={errors_dict.get("P")[d]}')
        plt.xlabel('grid')
        plt.ylabel(r'$L_2$(p)')
        plt.legend()
        origins_y = np.exp((np.log(errors_dict.get('L2_error').get('p')[-2,:]) + np.log(errors_dict.get('L2_error').get('p')[-1,:]))/2)
        origin_x = np.exp((np.log(16)+np.log(32))/2)
        fig = plt.gcf()
        draw_loglog_slope(fig, ax, (origin_x,origins_y[0]+origins_y[0]/3), -.23, -2, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[1]+origins_y[1]/3), -.23, -3, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[2]+origins_y[2]/3), -.23, -4, color='black')
        draw_loglog_slope(fig, ax, (origin_x,origins_y[3]+origins_y[3]/3), -.23, -5, color='black')

        if export:
            plt.savefig(rf'./postprocessing/plots/grid_convergence/grid_convergence_Stokes_p_{shape}_L2.svg', bbox_inches='tight')
        else:
            plt.show()

def plot_spectral_radius(sr_dict, export=False, filename='spectral_radius_Poisson'):
    plt.style.use(['science'])
    plt.rcParams.update(rc_params)
    plt.figure() if export else plt.figure(dpi=150)

    plt.loglog(sr_dict.get('grids'), np.ones(len(sr_dict['grids'])), '-k', label=r'$\rho($\textbf{\textit{B}}$)=1$')
    i=0
    for key, data in sr_dict['rectangle_sigmamul1'].items():
        plt.plot(sr_dict.get('grids'), np.array(data),  '--k', label=f'p={key}', marker=markers[i])
        i+=1
    ax = plt.gca()
    tick_locations = sr_dict.get('grids')
    tick_labels =  [f"{n}X{n}" for n in tick_locations]
    ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
    ax.set_xticklabels(tick_labels)
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_tick_params(which='minor', width=0) 
    plt.xlabel('grid')
    plt.ylabel(r'$\rho($\textbf{\textit{B}}$)$')
    plt.legend()
    if export:
        plt.savefig(rf'./postprocessing/plots/spectral_radius/{filename}_rectangle_sigmamul1.svg', bbox_inches='tight')
    else:
        plt.show()
    
    plt.style.use(['science'])
    plt.rcParams.update(rc_params)
    plt.figure() if export else plt.figure(dpi=150)
    plt.plot(sr_dict['grids'], np.ones(len(sr_dict['grids'])), '-k', label=r'$\rho($\textbf{\textit{B}}$)=1$')
    i=0
    for key, data in sr_dict['circle_sigmamul1'].items():
        plt.plot(sr_dict.get('grids'), np.array(data),  '--k', label=f'p={key}', marker=markers[i])
        i+=1
    ax = plt.gca()
    tick_locations = sr_dict.get('grids')
    tick_labels =  [f"{n}X{n}" for n in tick_locations]
    ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
    ax.set_xticklabels(tick_labels)
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_tick_params(which='minor', width=0) 
    plt.ylim([0.75,10])
    plt.xlabel('grid')
    plt.ylabel(r'$\rho($\textbf{\textit{B}}$)$')
    plt.legend()
    if export:
        plt.savefig(rf'./postprocessing/plots/spectral_radius/{filename}_circle_sigmamul1.svg', bbox_inches='tight')
    else:
        plt.show()

    plt.style.use(['science'])
    plt.rcParams.update(rc_params)
    plt.figure() if export else plt.figure(dpi=150)
    plt.plot(sr_dict['grids'], np.ones(len(sr_dict['grids'])), '-k', label=r'$\rho($\textbf{\textit{B}}$)=1$')
    i=0
    for key, data in sr_dict['circle_sigmamul2'].items():
        plt.plot(sr_dict.get('grids'), np.array(data),  '--k', label=f'p={key}', marker=markers[i])
        i+=1
    ax = plt.gca()
    tick_locations = sr_dict.get('grids')
    tick_labels =  [f"{n}X{n}" for n in tick_locations]
    ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
    ax.set_xticklabels(tick_labels)
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_tick_params(which='minor', width=0) 
    plt.xlabel('grid')
    plt.ylabel(r'$1-\rho($\textbf{\textit{B}}$)$')
    plt.legend()
    if export:
        plt.savefig(rf'./postprocessing/plots/spectral_radius/{filename}_circle_sigmamul2.svg', bbox_inches='tight')
    else:
        plt.show()
    
    plt.style.use(['science'])
    plt.rcParams.update(rc_params)
    plt.figure() if export else plt.figure(dpi=150)
    plt.plot(sr_dict['grids'], np.ones(len(sr_dict['grids'])), '-k', label=r'$\rho($\textbf{\textit{B}}$)=1$')
    i=0
    for key, data in sr_dict['circle_sigmamul1_ortho'].items():
        plt.plot(sr_dict.get('grids'), np.array(data),  '--k', label=f'p={key}', marker=markers[i])
        i+=1
    ax = plt.gca()
    tick_locations = sr_dict.get('grids')
    tick_labels =  [f"{n}X{n}" for n in tick_locations]
    ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
    ax.set_xticklabels(tick_labels)
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_tick_params(which='minor', width=0) 
    plt.ylim([0.75,10])
    plt.xlabel('grid')
    plt.ylabel(r'$\rho($\textbf{\textit{B}}$)$')
    plt.legend()
    if export:
        plt.savefig(rf'./postprocessing/plots/spectral_radius/{filename}_circle_sigmamul1_ortho.svg', bbox_inches='tight')
    else:
        plt.show()


    plt.style.use(['science'])
    plt.rcParams.update(rc_params)
    plt.figure() if export else plt.figure(dpi=150)
    # plt.loglog(sr_dict.get('grids'), np.ones(len(sr_dict['grids'])), '-k', label=r'$\rho($\textbf{\textit{B}}$)=1$')
    i=0
    plt.plot([], [], '--', label=f'Cartesian grid', color='black')
    plt.plot([], [], ':', label=f'Curvilinear grid', color='black')
    for (key_rectangle, data_rectangle), (key_circle, data_circle) in zip(sr_dict['rectangle_sigmamul1'].items(),sr_dict['circle_sigmamul2'].items()):
        plt.loglog(sr_dict.get('grids'), 1.-np.array(data_rectangle),  '--k', marker=markers[i])
        plt.loglog(sr_dict.get('grids'), 1.-np.array(data_circle),  ':k', marker=markers[i])
        plt.plot([], [], markers[i], label=f'p={key_rectangle}', color='black')
        i+=1
    ax = plt.gca()
    fig = plt.gcf()
    origins_y = np.exp((np.log(1-sr_dict.get('rectangle_sigmamul1').get('1')[-2]) + np.log(1-sr_dict.get('rectangle_sigmamul1').get('1')[-1]))/2)
    origin_x = np.exp((np.log(16)+np.log(32))/2)
    draw_loglog_slope(fig, ax, (origin_x,origins_y+origins_y/8), -.2, -2, color='black')
    tick_locations = sr_dict.get('grids')
    tick_labels =  [f"{n}X{n}" for n in tick_locations]
    ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
    ax.set_xticklabels(tick_labels)
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_tick_params(which='minor', width=0) 
    plt.xlabel('grid')
    plt.ylim([1e-4, 1])
    plt.ylabel(r'$1-\rho($\textbf{\textit{B}}$_{GS})$')
    legend = plt.legend()
    for t in legend.get_texts():
        t.set_ha('right')
    if export:
        plt.savefig(rf'./postprocessing/plots/spectral_radius/{filename}_rectangle_circle_SPD.svg', bbox_inches='tight')
    else:
        plt.show()

def plot_multigrid_results(residuals_list, export=False, filename=''):
    plt.style.use(['science'])
    plt.rcParams.update(rc_params)
    plt.rcParams.update({'text.latex.preamble': r'\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}'})
    plt.figure() if export else plt.figure(dpi=150)

    i=0
    n = len(residuals_list)-1
    for residuals_dict in residuals_list:
        cycles = range(1,len(residuals_dict.get('residuals'))+1)
        plt.semilogy(cycles, residuals_dict.get('residuals'),  'k', linestyle=lines[n-i], label=f'{residuals_dict.get("grid")}, p={residuals_dict.get("p")}')
        i+=1
    ax = plt.gca()
    # if len(cycles)<20:
    #     tick_locations = range(len(residuals_dict.get('residuals')))
    #     tick_labels =  range(len(residuals_dict.get('residuals')))
    #     print(tick_labels)
    #     ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_locations))
    #     ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locations))
    #     ax.set_xticklabels(tick_labels)
    #     ax.get_xaxis().set_tick_params(which='minor', size=0)
    #     ax.get_xaxis().set_tick_params(which='minor', width=0) 
    plt.ylim([1e-6, 1e1])
    # plt.xlim([0, plt.xlim()[1]])
    plt.xlabel('Number of V-cycles')
    plt.ylabel(r'$L_2(\mathcal{R})$')
    plt.legend()
    if export:
        plt.savefig(rf'./postprocessing/plots/multigrid/multigrid_cycles_{filename}_.svg', bbox_inches='tight')
    else:
        plt.show()

def plot_smoother_results(residuals_lists, export=False, filename=''):
    plt.style.use(['science'])
    plt.rcParams.update(rc_params)
    plt.rcParams.update({'figure.figsize': (3.3, 2.5), 'text.latex.preamble': r'\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}'})
    plt.figure(figsize=(3.3,2.5)) if export else plt.figure(figsize=(3.3,2.5), dpi=150)

    i=0
    n = len(residuals_lists)-1
    for residuals_list in residuals_lists:
        j=0
        for residuals_dict in residuals_list:
            iterations = range(1,len(residuals_dict.get('residuals'))+1)
            # plt.semilogy(iterations, residuals_dict.get('residuals'),  'k', linestyle=lines[n-i], label=f'{residuals_dict.get("grid")}, p_u={residuals_dict.get("p")}')
            if 'lsq' in filename:
                plt.semilogy(iterations, residuals_dict.get('residuals'), 'k', linestyle=lines[n-i], label=rf'{residuals_dict.get("grid")}, $p$={residuals_dict.get("p")}', marker=markers[j], markevery=0.2)#len(residuals_dict.get('residuals'))/10)
            else:
                plt.semilogy(iterations, residuals_dict.get('residuals'), 'k', linestyle=lines[n-i], label=rf'{residuals_dict.get("grid")}, $p$={residuals_dict.get("p")}')#, marker=markers[j], markevery=0.2)#len(residuals_dict.get('residuals'))/10)
                i += 1
                continue
            j += 1
        i+=1
    ax = plt.gca()
    # fig = plt.gcf()
    # fig.subplots_adjust(right=.5)
    if 'lsq' in filename:
        plt.ylim([1e-6, 1e1])
    else:
        plt.ylim([1e-1, 1e10])
    # plt.xlim([0, plt.xlim()[1]])
    plt.xlabel('Number of iterations')
    plt.ylabel(r'$L_2(\mathcal{R})$')
    # plt.legend()
    if 'lsq' in filename:
        plt.legend(bbox_to_anchor = (1.42, 0.5), loc='center right')
    else:
        plt.legend()
    if export:
        plt.savefig(rf'./postprocessing/plots/smoother/Stokes_smoother_{filename}_.svg', bbox_inches='tight')
    else:
        plt.show()



def draw_loglog_slope(fig, ax, origin, width_inches, slope, inverted=False, color=None, polygon_kwargs=None, label=False, labelcolor=None, label_kwargs=None, zorder=None):
    """
    This function draws slopes or "convergence triangles" into loglog plots.

    @param fig: The figure
    @param ax: The axes object to draw to
    @param origin: The 2D origin (usually lower-left corner) coordinate of the triangle
    @param width_inches: The width in inches of the triangle
    @param slope: The slope of the triangle, i.e. order of convergence
    @param inverted: Whether to mirror the triangle around the origin, i.e. whether 
        it indicates the slope towards the lower left instead of upper right (defaults to false)
    @param color: The color of the of the triangle edges (defaults to default color)
    @param polygon_kwargs: Additional kwargs to the Polygon draw call that creates the slope
    @param label: Whether to enable labeling the slope (defaults to false)
    @param labelcolor: The color of the slope labels (defaults to the edge color)
    @param label_kwargs: Additional kwargs to the Annotation draw call that creates the labels
    @param zorder: The z-order value of the triangle and labels, defaults to a high value
    """

    if polygon_kwargs is None:
        polygon_kwargs = {}
    if label_kwargs is None:
        label_kwargs = {}

    if color is not None:
        polygon_kwargs["color"] = color
    if "linewidth" not in polygon_kwargs:
        polygon_kwargs["linewidth"] = 0.75 * plt.rcParams["lines.linewidth"]
    if labelcolor is not None:
        label_kwargs["color"] = labelcolor
    if "color" not in label_kwargs:
        label_kwargs["color"] = polygon_kwargs["color"]
    if "fontsize" not in label_kwargs:
        label_kwargs["fontsize"] = 0.75 * plt.rcParams["font.size"]

    if inverted:
        width_inches = -width_inches
    if zorder is None:
        zorder = 10

    # For more information on coordinate transformations in Matplotlib see
    # https://matplotlib.org/3.1.1/tutorials/advanced/transforms_tutorial.html

    # Convert the origin into figure coordinates in inches
    origin_disp = ax.transData.transform(origin)
    origin_dpi = fig.dpi_scale_trans.inverted().transform(origin_disp)

    # Obtain the bottom-right corner in data coordinates
    corner_dpi = origin_dpi + width_inches * np.array([1.0, 0.0])
    corner_disp = fig.dpi_scale_trans.transform(corner_dpi)
    corner = ax.transData.inverted().transform(corner_disp)
    corner[0] += 13

    (x1, y1) = (origin[0], origin[1])
    x2 = corner[0]

    # The width of the triangle in data coordinates
    width = x2 - x1
    # Compute offset of the slope
    log_offset = y1 / (x1 ** slope)

    y2 = log_offset * ((x1 + width) ** slope)
    height = y2 - y1

    # The vertices of the slope
    a = origin
    b = corner
    c = [x2, y2]

    # Draw the slope triangle
    X = np.array([a, b, c])
    triangle = plt.Polygon(X[:3,:], fill=False, zorder=zorder, **polygon_kwargs)
    ax.add_patch(triangle)

    # # Convert vertices into display space
    # a_disp = ax.transData.transform(a)
    # b_disp = ax.transData.transform(b)
    # c_disp = ax.transData.transform(c)

    # # Figure out the center of the triangle sides in display space
    # bottom_center_disp = a_disp + 0.5 * (b_disp - a_disp)
    # bottom_center = ax.transData.inverted().transform(bottom_center_disp)

    # right_center_disp = b_disp + 0.5 * (c_disp - b_disp)
    # right_center = ax.transData.inverted().transform(right_center_disp)

    # # Label alignment depending on inversion parameter
    # va_xlabel = "top" if not inverted else "bottom"
    # ha_ylabel = "left" if not inverted else "right"

    # # Label offset depending on inversion parameter
    # offset_xlabel = [0.0, -0.33 * label_kwargs["fontsize"]] if not inverted else [0.0, 0.33 * label_kwargs["fontsize"]]
    # offset_ylabel = [0.33 * label_kwargs["fontsize"], 0.0] if not inverted else [-0.33 * label_kwargs["fontsize"], 0.0]

    # # Draw the slope labels
    # ax.annotate("$1$", bottom_center, xytext=offset_xlabel, textcoords='offset points', ha="center", va=va_xlabel, zorder=zorder, **label_kwargs)
    # ax.annotate(f"${slope}$", right_center, xytext=offset_ylabel, textcoords='offset points', ha=ha_ylabel, va="center", zorder=zorder, **label_kwargs)