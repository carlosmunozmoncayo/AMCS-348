from scipy.special import legendre as legendre_poly
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt


#x=float(input("insert x: "))
#cubic=legendre_poly(3)
#print(cubic(x))
#print(2.5*x**3-1.5*x)

def generate_lagrange_poly(j, x_nodes='LGL',n=10):
    if x_nodes == 'LGL':
        _,_,_,_,_,_,x_nodes,_= generate_LGL_points(n-1)
    elif x_nodes == 'LG':
        _,_,_,_,x_nodes,_,_,_= generate_LGL_points(n-1)
    xj = x_nodes[j]
    n_nodes = len(x_nodes)
    def Lj(x):
        #Returns Lagrange polynomial
        #L_j(x)=\prod_{k \neq j} \frac{x-x_k}{x_j-x_k}
        mask = np.invert(np.eye(n_nodes, dtype = bool)[j])
        return np.prod((x-x_nodes)/(xj-x_nodes), where = mask)

    def Ljp(x):
        #Returns first derivative of Lj
        #L_j^{\prime}(x)=\frac{\sum_{k=0, k \neq j}^n \prod_{l=0, l \neq k, l \neq j}^n\left(x-x_l\right)}{\prod_{k=0, k \neq j}^n\left(x_j-x_k\right)}
        suma = 0
        for k in range(n_nodes):
            mask = np.ones(n_nodes,dtype=bool); mask[k]=False; mask[j]=False
            if k != j:
                suma += np.prod(x-x_nodes, where = mask)
        mask = np.ones(n_nodes,dtype=bool); mask[j]=False
        return suma/np.prod(xj-x_nodes, where = mask)

    return Lj, Ljp

def generate_LGL_points(n):
    #legendre_poly(n) returns the L_k Legendre polynomial, i.e.,
    #((1-x^2) L_k'(x))'+k(k+1) L_k(x) = 0
    p_Legn = legendre_poly(n) #Nth Legendre polynomial 
    p_Legn_prime = np.polyder(p_Legn) #Derivative of Nth Legendre polynomial
    vp_Legn = np.vectorize(p_Legn)

    p_Legn1 = legendre_poly(n+1) #N+1th Legendre polynomial
    p_Legn1_prime = np.polyder(p_Legn1) #Derivative of N+1th Legendre polynomial
    vp_Legn1_prime = np.vectorize(p_Legn1_prime) #Vectorizing previous function

    x_LG = np.roots(p_Legn1) #Legendre-Gauss points
    w_LG = 2/(1.-x_LG**2)/(vp_Legn1_prime(x_LG))**2 #Legendre-Gauss weights

    x_LGL = np.roots(p_Legn_prime)
    x_LGL = np.append(-1., x_LGL)
    x_LGL = np.append(x_LGL, 1.)
    w_LGL = 2./(n*(n+1))/vp_Legn(x_LGL)**2
        
    return (p_Legn, p_Legn_prime,
            p_Legn1, p_Legn1_prime,
            x_LG, w_LG,
            x_LGL, w_LGL)

