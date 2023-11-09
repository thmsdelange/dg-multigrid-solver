from time import perf_counter_ns

class Timer:
    def __init__(self, logger=None):
        self.start_time = None
        self.end_time = None
        self.logger = logger

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self.logger is not None:
            self.logger.debug(f"Elapsed time: {self.elapsed():.4g} seconds")
    
    def start(self):
        self.start_time = perf_counter_ns()
    
    def stop(self):
        self.end_time = perf_counter_ns()
    
    def elapsed(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        elif self.end_time is None:
            return (perf_counter_ns() - self.start_time)/1e9
        else:
            return (self.end_time - self.start_time)/1e9

    def timeit(self, func):
        def wrapper(*args, **kwargs):
            self.start()
            result = func(*args, **kwargs)
            self.stop()
            if self.logger is not None:
                self.logger.debug(f"{func.__qualname__.split('.')[0]}.{func.__name__} took {self.elapsed():.4g} seconds")
            return result
        return wrapper