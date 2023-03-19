import numpy as np
from basis_nodes import generate_lagrange_poly, generate_LGL_points

def first_order_D( x_nodes='LGL' , n=10 ):
    if x_nodes == 'LGL':
        _,_,_,_,_,_,x_nodes,_ = generate_LGL_points(n-1)
    elif x_nodes == 'LG':
        _,_,_,_,x_nodes,_,_,_ = generate_LGL_points(n-1)

    n_nodes = len(x_nodes)
    D = np.zeros((n_nodes,n_nodes))
    for j in range(n_nodes):
        _,Ljp = generate_lagrange_poly(j=j,x_nodes=x_nodes,n=n_nodes)
        for i in range(n_nodes):
            D[i,j] = Ljp(x_nodes[i])
    return D

def first_order_P_Q( x_Lagrange_nodes='LGL', x_abcissae='LGL' , n=10 ):
    if x_Lagrange_nodes == 'LGL':
        _,_,_,_,_,_,x_Lagrange_nodes,_ = generate_LGL_points(n-1)

    n_nodes = len(x_Lagrange_nodes)
    
    if x_abcissae == 'LGL':
        _,_,_,_,_,_,x_abcissae, w_abcissae = generate_LGL_points(n_nodes-1)

    _,_,_,_,_,_,_, w_abcissae = generate_LGL_points(n_nodes-1)
    w_abcissae = 0.5*(x_abcissae[-1]-x_abcissae[0])*w_abcissae

    #Generating a list of Lagrange basis
    list_Lagrange_poly = []
    list_Lagrange_poly_prime = []
    for j in range(n_nodes):
        Lj ,Ljp = generate_lagrange_poly(j=j,x_nodes=x_Lagrange_nodes,n=n_nodes)
        list_Lagrange_poly.append(Lj)
        list_Lagrange_poly_prime.append(Ljp)

    #Filling matrix P
    P = np.zeros((n_nodes, n_nodes))
    for l in range(n_nodes):
        L_vec = np.array([ Lagrange_poly(x_abcissae[l]) for Lagrange_poly in list_Lagrange_poly])
        P += w_abcissae[l]*np.outer(L_vec,L_vec)

    #Filling matrix Q (It's faster to build it as Q=PD if we have D)
    #This could be done in the previous loop, but in practice we wouldn't even 
    #compute Q like this
    Q = np.zeros((n_nodes,n_nodes))
    for l in range(n_nodes):
        L_vec = np.array([ Lagrange_poly(x_abcissae[l]) for Lagrange_poly in list_Lagrange_poly])
        Lp_vec = np.array([ Lagrange_poly_prime(x_abcissae[l]) for Lagrange_poly_prime in list_Lagrange_poly_prime])
        Q += w_abcissae[l]*np.outer(L_vec, Lp_vec)

    return P,Q
        
