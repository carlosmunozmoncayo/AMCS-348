import numpy as np
import matplotlib.pyplot as plt

#Local libraries
from basis_nodes import generate_lagrange_poly, generate_LGL_points
from SBP_matrices import first_order_D, first_order_P_Q

#Automating what is done in the Jupiter notebooks

def generate_RHS_1D_advection(xlower=0.,xupper=2.,n_elements=40,
                                order_x=4, c_advection=1., BC="periodic",
                                num_flux_type="upwind"):
    #We define an array with the element boundary nodes
    x = np.linspace(xlower,xupper, n_elements+1)
    #We'll need a Lagrange basis of degree order_x-1
    #Thus, we will need order_x nodes in each element
    degree_basis = order_x-1
    if degree_basis >0:
        (_, _,
        _, _,
        _, _,
        xi_LGL_unsorted, w_LGL_unsorted) = generate_LGL_points(degree_basis)
    else:
        xi_LGL_unsorted, w_LGL_unsorted = np.array([0.]), np.array([1.])

    #######################
    #######################
    #The root finder returns unsorted roots,
    #We fix that in these lines

    #Indexes that would sort x_LG using value
    sort_idxs_LGL = np.argsort(xi_LGL_unsorted)
    #Sorting x_LG and w_LG using those indexes
    xi_LGL = xi_LGL_unsorted[sort_idxs_LGL]
    w_LGL = w_LGL_unsorted[sort_idxs_LGL]
    #######################
    #######################

    n_nodes = len(xi_LGL)

    #We map the nodes from the reference element to one of the elements 
    #from our computational domain (this suffices since we are using an uniform grid)
    x_min = x[0]
    x_max = x[1]

    #We just need to map the LGL nodes once since we are using a uniform grid
    x_element = (xi_LGL*(x_max-x_min)+(x_min+x_max))/2
    len_element = (x_max-x_min)/2. #Constant in this case
    w_element = (x_max-x_min)*w_LGL
    list_elements = [x_element]

    #We define our grid (interface points will be repeated)
    x_grid = np.zeros(n_elements*n_nodes)

    #Local opretaros P and Q
    P_LGL, Q_LGL = first_order_P_Q(x_Lagrange_nodes=xi_LGL, x_abcissae=xi_LGL, w_abcissae=w_LGL)

    #Global operators P and Q
    P = np.kron(np.eye(n_elements), P_LGL)
    Q = np.kron(np.eye(n_elements), Q_LGL)

    diagP=np.diag(P)
    diagPinv=1./diagP

    #Restriction operators R, B
    #Local
    R_LGL = np.zeros((2,n_nodes))
    R_LGL[0,0] = R_LGL[-1,-1] = 1
    B_LGL = np.zeros((2,2))
    B_LGL[0,0] = -1; B_LGL[-1,-1]=1
    #Global
    R = np.kron(np.eye(n_elements), R_LGL)
    B = np.kron(np.eye(n_elements), B_LGL)


    #Differential operator D
    #Local
    D_LGL = first_order_D(x_nodes=xi_LGL)
    #Global
    D = np.kron(np.eye(n_elements), D_LGL)


    #Vector and matrix with advection speed
    c_vec = c_advection*np.ones_like(x_grid)
    c_mat = c_advection*np.eye(len(x_grid))

    #Numerical flux
    #Local
    if num_flux_type == "upwind":
        f_num_loc = lambda uL, uR : c_advection*uL #Upwind flux
    elif num_flux_type == "centered:"
        f_num_loc = lambda uL, uR : c_advection*(uL+uR)/2 #Centered flux
    
    #Global
    def f_num(u,n_elements=n_elements,n_nodes=n_nodes):
        #Computing numerical flux just at the interface of elements
        #Fill numerical flux vector if there is just one element
        if n_elements == 1:
            f = [c*u[-1], c*u[-1]]  #The first component should be c*source at the left
                                    #Here we are using periodic BCs
            return np.array(f)

        #Fill numerical flux vector if we have more elements
        #This is coded for periodic BCs
        #Fill numerical flux vector for first element
        f = [f_num_loc(u[-1],u[0]),         #Left interface
            f_num_loc(u[n_nodes-1],u[n_nodes])]  #Right interface
        for idx_elem in range(1,n_elements-1):
            #Left interface
            idx_R = idx_elem*n_nodes
            idx_L = idx_R-1
            f.append(f_num_loc(u[idx_L],u[idx_R]))
            idx_R = (idx_elem+1)*n_nodes
            idx_L = idx_R-1
            f.append(f_num_loc(u[idx_L],u[idx_R]))
            
        #Fill numerical flux vector for last element
        f.append(c*u[-n_nodes-1]) #Left interface
        f.append(c*u[-1])       #Right interface
                           
        return np.array(f)

    def RHS_fun(u):
        split_form_interior = -0.5*D@c_mat@u -0.5*c_mat@D@u -0.5*np.diag(u)@D@c_vec
        non_split_form_interior = -c*D@u
        elem_boundary_terms = -diagPinv*(R.T@B@(f_num(u=u)-c*R@u))
        return (1./len_element)*(non_split_form_interior+elem_boundary_terms)
    
    return RHS_fun, x_grid, xi_LGL

