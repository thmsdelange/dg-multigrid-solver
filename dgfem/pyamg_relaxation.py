from warnings import warn

import numpy as np
from scipy import sparse
from scipy.linalg import lapack as la

from pyamg.util.utils import type_prep, get_diagonal, get_block_diag
from pyamg.util.params import set_tol
from pyamg.util.linalg import norm
from pyamg import amg_core


def make_system(A, x, b, formats=None):
    """Return A,x,b suitable for relaxation or raise an exception.

    Parameters
    ----------
    A : sparse-matrix
        n x n system
    x : array
        n-vector, initial guess
    b : array
        n-vector, right-hand side
    formats: {'csr', 'csc', 'bsr', 'lil', 'dok',...}
        desired sparse matrix format
        default is no change to A's format

    Returns
    -------
    (A,x,b), where A is in the desired sparse-matrix format
    and x and b are "raveled", i.e. (n,) vectors.

    Notes
    -----
    Does some rudimentary error checking on the system,
    such as checking for compatible dimensions and checking
    for compatible type, i.e. float or complex.

    Examples
    --------
    >>> from pyamg.relaxation.relaxation import make_system
    >>> from pyamg.gallery import poisson
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> (A,x,b) = make_system(A,x,b,formats=['csc'])
    >>> print(x.shape)
    (100,)
    >>> print(b.shape)
    (100,)
    >>> print(A.format)
    csc

    """
    if formats is None:
        pass
    elif formats == ['csr']:
        if sparse.isspmatrix_csr(A):
            pass
        elif sparse.isspmatrix_bsr(A):
            A = A.tocsr()
        else:
            warn('implicit conversion to CSR', sparse.SparseEfficiencyWarning)
            A = sparse.csr_matrix(A)
    else:
        if sparse.isspmatrix(A) and A.format in formats:
            pass
        else:
            A = sparse.csr_matrix(A).asformat(formats[0])

    if not isinstance(x, np.ndarray):
        raise ValueError('expected numpy array for argument x')
    if not isinstance(b, np.ndarray):
        raise ValueError('expected numpy array for argument b')

    M, N = A.shape

    if M != N:
        raise ValueError('expected square matrix')

    if x.shape not in [(M,), (M, 1)]:
        raise ValueError('x has invalid dimensions')
    if b.shape not in [(M,), (M, 1)]:
        raise ValueError('b has invalid dimensions')

    if A.dtype != x.dtype or A.dtype != b.dtype:
        raise TypeError('arguments A, x, and b must have the same dtype')

    if not x.flags.carray:
        raise ValueError('x must be contiguous in memory')

    x = np.ravel(x)
    b = np.ravel(b)

    return A, x, b

def gauss_seidel(A, x, b, iterations=1, sweep='forward'):
    """Perform Gauss-Seidel iteration on the linear system Ax=b.

    Parameters
    ----------
    A : csr_matrix, bsr_matrix
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    iterations : int
        Number of iterations to perform
    sweep : {'forward','backward','symmetric'}
        Direction of sweep

    Returns
    -------
    Nothing, x will be modified in place.

    Examples
    --------
    >>> # Use Gauss-Seidel as a Stand-Alone Solver
    >>> from pyamg.relaxation.relaxation import gauss_seidel
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x0 = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> gauss_seidel(A, x0, b, iterations=10)
    >>> print(f'{norm(b-A*x0):2.4}')
    4.007
    >>> #
    >>> # Use Gauss-Seidel as the Multigrid Smoother
    >>> from pyamg import smoothed_aggregation_solver
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...         coarse_solver='pinv', max_coarse=50,
    ...         presmoother=('gauss_seidel', {'sweep':'symmetric'}),
    ...         postsmoother=('gauss_seidel', {'sweep':'symmetric'}))
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)

    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])

    if sparse.isspmatrix_csr(A):
        blocksize = 1
    else:
        R, C = A.blocksize
        if R != C:
            raise ValueError('BSR blocks must be square')
        blocksize = R

    if sweep not in ('forward', 'backward', 'symmetric'):
        raise ValueError('valid sweep directions: "forward", "backward", and "symmetric"')

    if sweep == 'forward':
        row_start, row_stop, row_step = 0, int(len(x)/blocksize), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = int(len(x)/blocksize)-1, -1, -1
    elif sweep == 'symmetric':
        for _iter in range(iterations):
            gauss_seidel(A, x, b, iterations=1, sweep='forward')
            gauss_seidel(A, x, b, iterations=1, sweep='backward')
        return

    if sparse.isspmatrix_csr(A):
        for _iter in range(iterations):
            amg_core.gauss_seidel(A.indptr, A.indices, A.data, x, b,
                                  row_start, row_stop, row_step)
    else:
        for _iter in range(iterations):
            amg_core.bsr_gauss_seidel(A.indptr, A.indices, np.ravel(A.data),
                                      x, b, row_start, row_stop, row_step, R)

def block_gauss_seidel(A, x, b, iterations=1, sweep='forward', blocksize=1, Dinv=None):
    """Perform block Gauss-Seidel iteration on the linear system Ax=b.

    Parameters
    ----------
    A : csr_matrix, bsr_matrix
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    iterations : int
        Number of iterations to perform
    sweep : {'forward','backward','symmetric'}
        Direction of sweep
    Dinv : array
        Array holding block diagonal inverses of A
        size (N/blocksize, blocksize, blocksize)
    blocksize : int
        Desired dimension of blocks


    Returns
    -------
    Nothing, x will be modified in place.

    Examples
    --------
    >>> # Use Gauss-Seidel as a Stand-Alone Solver
    >>> from pyamg.relaxation.relaxation import block_gauss_seidel
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> x0 = np.zeros((A.shape[0],1))
    >>> b = np.ones((A.shape[0],1))
    >>> block_gauss_seidel(A, x0, b, iterations=10, blocksize=4, sweep='symmetric')
    >>> print(f'{norm(b-A*x0):2.4}')
    0.9583
    >>> #
    >>> # Use Gauss-Seidel as the Multigrid Smoother
    >>> from pyamg import smoothed_aggregation_solver
    >>> opts = {'sweep':'symmetric', 'blocksize' : 4}
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...        coarse_solver='pinv', max_coarse=50,
    ...        presmoother=('block_gauss_seidel', opts),
    ...        postsmoother=('block_gauss_seidel', opts))
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)

    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])
    A = A.tobsr(blocksize=(blocksize, blocksize))

    if Dinv is None:
        Dinv = get_block_diag(A, blocksize=blocksize, inv_flag=True)
    elif Dinv.shape[0] != int(A.shape[0]/blocksize):
        raise ValueError('Dinv and A have incompatible dimensions')
    elif (Dinv.shape[1] != blocksize) or (Dinv.shape[2] != blocksize):
        raise ValueError('Dinv and blocksize are incompatible')

    if sweep not in ('forward', 'backward', 'symmetric'):
        raise ValueError('valid sweep directions: "forward", "backward", and "symmetric"')

    if sweep == 'forward':
        row_start, row_stop, row_step = 0, int(len(x)/blocksize), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = int(len(x)/blocksize)-1, -1, -1
    elif sweep == 'symmetric':
        for _iter in range(iterations):
            block_gauss_seidel(A, x, b, iterations=1, sweep='forward',
                            blocksize=blocksize, Dinv=Dinv)
            block_gauss_seidel(A, x, b, iterations=1, sweep='backward',
                            blocksize=blocksize, Dinv=Dinv)
        return

    for _iter in range(iterations):
        amg_core.block_gauss_seidel(A.indptr, A.indices, np.ravel(A.data),
                                    x, b, np.ravel(Dinv),
                                    row_start, row_stop, row_step, blocksize)