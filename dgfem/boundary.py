### this class can in the future be extended to apply e.g. Dirichlet or Neumann boundary conditions (right now this is not needed for the MMS solutions)

class Boundary:
    def __init__(self, which):
        self.which = which

    def dirichlet_boundary(self, value):
        pass
    
    def neumann_boundary(self):
        pass