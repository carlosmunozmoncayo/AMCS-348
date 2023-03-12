from scipy.special import legendre as legendre_poly
import numpy as np
import matplotlib.pyplot as plt

#legendre_poly(n) returns the L_k Legendre polynomial, i.e.,
#((1-x^2) L_k'(x))'+k(k+1) L_k(x) = 0

#x=float(input("insert x: "))
cubic=legendre_poly(3)
print(cubic(x))
print(2.5*x**3-1.5*x)

def generate_lagrange_poly(j, x_nodes='LGL'):
    xj = x_nodes[j]
    n_nodes = len(x_nodes)
    def Lj(x):
        #Returns Lagrange polynomial
        #L_j(x)=\prod_{k \neq j} \frac{x-x_k}{x_j-x_k}
        mask = np.invert(np.eye(n, dtype = bool)[j])
        return np.prod((x-x_nodes)/(xj-x_nodes), where = mask)

    def Ljp(x):
        #Returns first derivative of Lj
        #L_i(x)^{\prime}=\frac{\sum_{k=0}^n \prod_{l=0, l \neq k}^n\left(x-x_l\right)}{\prod_{k=0, k \neq j}^n\left(x_j-x_k\right)}        return
        suma = 0
        for k in range(n_nodes):
            mask = np.invert(np.eye(n, dtype = bool)[k])
            suma += np.prod(x-x_nodes, where = mask)
        mask = np.invert(np.eye(n, dtype = bool)[j])
        return suma/np.prod(xj-x_nodes, where = mask)

    return Lj, Ljp
