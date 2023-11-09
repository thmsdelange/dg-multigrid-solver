import numpy as np

def compute_m(i:int,j:int, Ni:int):
    """Compute the two-dimensional index m for an element with indices i,j

    Args:
        i (int): index 
        j (int): index
        Ni (int): number of elements in i-direction

    Returns:
        int: the index m
    """
    return int(j*Ni+i)

def compute_Lp_norm(delta: np.ndarray, p:int):
    """Compute the Lp norm given some delta and order p

    Args:
        delta (np.ndarray): difference between numerical and exact solution or difference between the right-hand side and the coefficient matrix multiplied by the numerical solution (residual)
        p (int): order

    Returns:
        np.float64: Lp norm of delta
    """
    return (np.sum(abs(delta)**p)/delta.size)**(1/p)

def compute_residual_norm(grid:'Grid', u:np.ndarray, p:int=2):
    """Compute the Lp norm of the residual

    Args:
        grid (Grid): Grid object
        u (np.ndarray): numerical solution
        p (int, optional): order, defaults to 2

    Returns:
        np.float64: Lp norm of the residual
    """
    return compute_Lp_norm(grid.RHS - grid.BSR @ u, p)

def reorder_global_to_local_DOFs(grid:'Grid', vector:np.ndarray) -> np.ndarray: 
    """Reorder a vector from global to local ordering

    Args:
        grid (Grid): Grid object
        vector (np.ndarray): vector that needs to be reordered

    Returns:
        np.ndarray: reordered vector
    """
    indices = np.arange(grid.N*grid.N_DOF_sol_tot)
    indices_i = []
    for i in range(grid.N):
        indices_i.extend(indices[i*grid.N_DOF_sol.get('u'):(i+1)*grid.N_DOF_sol.get('u')])
        indices_i.extend(indices[grid.N*grid.N_DOF_sol.get('u')+i*grid.N_DOF_sol.get('u'):grid.N*grid.N_DOF_sol.get('u')+(i+1)*grid.N_DOF_sol.get('u')])
        indices_i.extend(indices[grid.N*grid.N_DOF_sol.get('u')*2+i*grid.N_DOF_sol.get('p'):grid.N*grid.N_DOF_sol.get('u')*2+(i+1)*grid.N_DOF_sol.get('p')])

    return vector[indices_i]

def reorder_local_to_global_DOFs(grid:'Grid', vector:np.ndarray) -> np.ndarray:
    """Reorder a vector from local to global ordering

    Args:
        grid (Grid): Grid object
        vector (np.ndarray): vector that needs to be reordered

    Returns:
        np.ndarray: reordered vector
    """
    indices = np.arange(grid.N*grid.N_DOF_sol_tot)
    indices_u = []
    indices_v = []
    indices_p = []
    for i in range(grid.N):
        indices_u.extend(indices[i*grid.N_DOF_sol_tot:i*grid.N_DOF_sol_tot+grid.N_DOF_sol.get('u')])
        indices_v.extend(indices[i*grid.N_DOF_sol_tot+grid.N_DOF_sol.get('u'):i*grid.N_DOF_sol_tot+grid.N_DOF_sol.get('u')*2])
        indices_p.extend(indices[i*grid.N_DOF_sol_tot+2*grid.N_DOF_sol.get('u'):(i+1)*grid.N_DOF_sol_tot])
    
    indices = np.concatenate([np.array(indices_u),np.array(indices_v),np.array(indices_p)]).astype(int)
    return vector[indices]

def obj_to_dict(obj):
    if isinstance(obj, (int, str, float)):
        return obj
    elif isinstance(obj, list):
        return [obj_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: obj_to_dict(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {key: obj_to_dict(value) for key, value in obj.__dict__.items()}
    else:
        return str(obj)
    
def convert_to_dict(N, r, s):
    if not isinstance(N, dict):
        N = {'u': N}
    
    if not isinstance(r, dict) and not isinstance(s, dict):
        r = {'u': r}
        s = {'u': s}
    elif not isinstance(r, dict):
        r = {key: r for key in s.keys()}
    elif not isinstance(s, dict):
        s = {key: s for key in r.keys()}
    elif isinstance(r, dict) and isinstance(s, dict):
        if len(r.keys()) < len(s.keys()):
            r = {key: r for key in s.keys()}
        elif len(s.keys()) < len(r.keys()):
            s = {key: s for key in r.keys()}
    return N, r, s

def is_diagonally_dominant(x):
    abs_x = np.abs(x)
    return np.all( 2*np.diag(abs_x) >= np.sum(abs_x, axis=1) )


def compute_row_echelon(A: np.ndarray) -> np.ndarray:
    """Compute the row echelon form of matrix A

    Args:
        A (np.ndarray): matrix for which the row echelon form must be computed

    Returns:
        np.ndarray: row echelon form of A
    """

    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    A = A.copy()
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = compute_row_echelon(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]

    # we perform REF on matrix from second row, from second column
    B = compute_row_echelon(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])
