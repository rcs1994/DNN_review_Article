import numpy as np
from sympy import *

x1,x2 = symbols('x1 x2')
variables = [x1,x2]

def near(x,y,tolerance):
    if Or(x-y < tolerance, y-x < tolerance):
        return True
    else:
        return False

class gt_gen(object):
    def __init__(self):
        self.variable = variables

        '''self.f = -2.0
        self.bdry = Piecewise(
            (x1**2,And(And(0<=x1,x1<= 0.5),Or(near(x2,0,1e-8),near(x2,1,1e-8)))),
            ((x1-1)**2,And(And(x1>=0.5,x1<=1),Or(near(x2,0,1e-8),near(x2,1,1e-8)))),
            (0,True)
            )
        self.y = Piecewise(
            (x1**2,And(x1<=0.5,x1>=0.0)),
            ((x1-1)**2,True)
        )'''

        #self.f = -4
        #self.bdry = x1**2+x2**2
        #self.y = x1**2+x2**2
        self.f = -(((1 - 2*x1)**2 - 2 - pi**2) * exp(x1*(1-x1)) * sin(pi*x2)) -(((1 - 2*x2)**2 - 2 - pi**2) * exp(x2*(1-x2)) * sin(pi*x1))
        self.bdry = exp(x1*(1-x1))*sin(pi*x2) + exp(x2*(1-x2))*sin(pi*x1)
        self.y = exp(x1*(1-x1))*sin(pi*x2) + exp(x2*(1-x2))*sin(pi*x1)
        self.y_x1 = diff(self.y,x1)
        self.y_x2 = diff(self.y,x2)
        

    def generate(self,func,collocations):
        #shape of collocation: (N,2)
        ldfunc = lambdify(variables,func,'numpy')
        data = [
        ldfunc(d[0],d[1]) for d in collocations
        ]
        return np.array(data)



