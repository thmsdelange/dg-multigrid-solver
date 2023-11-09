class MutuallyInclusiveArgumentError(Exception):
        pass
class MutuallyExclusiveArgumentError(Exception):
        pass

if __name__ == '__main__':
    ### Top-level function to pass options to solver and execute
    import argparse
    import sys
    parser = argparse.ArgumentParser(
                    prog='DG solver',
                    description='DG solver for the Poisson and Stokes problems')
    parser.add_argument('--grid-folder', type=str)
    parser.add_argument('-f', '--grid-file', type=str)
    parser.add_argument('--p-grid', type=int)
    parser.add_argument('--p-solution', type=int)
    
    solver = parser.add_mutually_exclusive_group(required=True)
    solver.add_argument('-d', '--solve-direct', action='store_true')
    solver.add_argument('-s', '--solve-smoother', help='mutually inclusive with --smoother', action='store_true')
    parser.add_argument('--smoother', type=str)
        
    solver.add_argument('-amg', '--solve-pyamg', action='store_true')
    solver.add_argument('-k', '--solve-krylov', action='store_true')
    solver.add_argument('-m', '--solve-multigrid', action='store_true')  
    solver.add_argument('-fvm', '--solve-finite-volume-method', action='store_true')  
    
    solver.add_argument('-amp', '--solve-smoother-amplification', help='mutually inclusive with --fvm-discretization or --dg-discretization', action='store_true')
    parser.add_argument('--dg-discretization', action='store_true')
    parser.add_argument('--fvm-discretization', action='store_true')
    
    parser.add_argument('--check-eigenvalues', action='store_true')
    parser.add_argument('--check-condition-number', action='store_true')
    parser.add_argument('--plot-sparsity-pattern', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--silent', action='store_true')
    
    args = parser.parse_args()
    if args.solve_smoother and not args.smoother:
        raise MutuallyInclusiveArgumentError("--solve-smoother option must be used with --smoother" )
    
    discretization = None
    if args.solve_smoother_amplification:
        if not (args.dg_discretization or args.fvm_discretization):
            raise MutuallyInclusiveArgumentError("--solve-smoother-amplification option must be used with either --dg-discretization or --fvm-discretization" )
        if args.dg_discretization and args.fvm_discretization:
            raise MutuallyExclusiveArgumentError('--dg-discretization cannot be used together with --fvm-discretization')
        if args.dg_discretization:
            discretization = 'dg'
        elif args.fvm_discretization:
            discretization = 'fvm'

    from input import params
    from dgfem.settings import Settings
    settings = Settings(params)
    if args.verbose: settings.update_setting('logging.loglevel', 'DEBUG')
    if args.silent: settings.update_setting('logging.loglevel', 'ERROR')


    from dgfem.dgfem import DGFEM
    from utils.logger import Logger
    import traceback

    logger = Logger(__name__, settings).logger
    logger.info('starting DG-FEM')

    try:
        dgfem = DGFEM(settings=settings, grid_folder=args.grid_folder, grid_file=args.grid_file, p_grid=args.p_grid, p_solution=args.p_solution,
                      solve_direct=args.solve_direct, solve_smoother=args.solve_smoother, solve_smoother_amplification=args.solve_smoother_amplification, solve_pyamg=args.solve_pyamg, 
                      solve_krylov=args.solve_krylov, solve_multigrid=args.solve_multigrid, solve_finite_volume_method=args.solve_finite_volume_method, smoother=args.smoother, discretization=discretization,
                      check_eigenvalues=args.check_eigenvalues, check_condition_number=args.check_condition_number, plot_sparsity_pattern=args.plot_sparsity_pattern)
        dgfem.solve()
    except Exception:
        logger.critical(traceback.format_exc())