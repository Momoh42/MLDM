import numpy as np
from numpy import linalg as LA
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import itertools
import operator
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import os
import pickle
import matplotlib.pyplot as plt 
from scipy.stats import entropy

from operator import itemgetter

from tqdm import tqdm

import derivative
from pysindy.feature_library import FourierLibrary, CustomLibrary
from pysindy.feature_library import ConcatLibrary
from pysindy.feature_library import PolynomialLibrary
from sklearn.linear_model import HuberRegressor
import copy
"""
Added functions ML project
"""
def open_pkl_to_dicts(folder_path):
    list_of_dicts = []
    # List all pickle files in the folder
    pickle_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    # Iterate over each pickle file
    for pickle_file in pickle_files:
        file_path = os.path.join(folder_path, pickle_file)
        # Load the dictionary from the pickle file
        with open(file_path, 'rb') as file:
            loaded_dict = pickle.load(file)
        # Append the loaded dictionary to the list
        list_of_dicts.append(loaded_dict)
    return list_of_dicts
def get_array(list_of_dicts, key):
    list_arrays = []
    for di in list_of_dicts:
        list_arrays.append(np.array(di[key]))
    return list_arrays
def get_centered_patches(data_in_3d_array, size=200):
    data_in_3d_array = np.array(data_in_3d_array)
    center_y = data_in_3d_array[0].shape[0] // 2
    center_x = data_in_3d_array[0].shape[1] // 2
    halfsize= size//2
    data_patches = data_in_3d_array[:,center_y-(halfsize):center_y+(halfsize),
                                    center_x-(halfsize):center_x+(halfsize)]
def plt_images(list_of_images3Darrays, t_in_secs=[0,4,9,19,29,39,49,59],save_name=None):
    t = t_in_secs*8
    cols = len(t_in_secs)
    rows = len(list_of_images3Darrays) 
    fig, ax = plt.subplots(rows,cols, sharex=True,sharey=True)
    for r in range(rows):
        for c in range(cols):
            ax[r,c].imshow(list_of_images3Darrays[r][t[c]])

    # Save the plot if a file name is provided
    if save_name:
        plt.savefig(save_name)
        print(f"Plot saved as {save_name}")
        plt.show()
    else:
        # Display the plot
        plt.show()

def top_x_scores(CVdict_results, description, x:int=10):
    my_list = [i[-1] for i in CVdict_results['res_r2_score_history']]
    for i,j in enumerate(my_list):
        my_list[i] = (i,j)
    my_list = sorted(my_list, reverse=True, key=itemgetter(1))

    my_list = sorted(my_list, reverse=True, key=itemgetter(1))
    how_to_order = []
    for i,(id,val) in enumerate(my_list):
        my_list[i] = val
        how_to_order.append(id)

    lambda_order = [CVdict_results['lam_values'][i // CVdict_results['d_tol_values'].shape[0]] for i in how_to_order]
    d_tol_order = [CVdict_results['d_tol_values'][i % CVdict_results['d_tol_values'].shape[0]] for i in how_to_order]

    print(f"how_to_order : {how_to_order[:x]}")
    print(f"top_10_values : {my_list[:x]}")
    print(f"top_10_lambda : {lambda_order[:x]}")
    print(f"top_10_d_tol : {d_tol_order[:x]}")

    for i in range(len(how_to_order[:x])):
        r2_score = my_list[i]
        lam = lambda_order[i]
        d_tol = d_tol_order[i]
        print(f"r2_score = {r2_score}, lam = {lam}, d_tol = {d_tol}, id = {how_to_order[i]}")
        print_pde(CVdict_results['res_c'][how_to_order[i]], description)


def get_grid(CVdict_results):
    """
    output the grid of r2 scores based on changing lambda and d_tol on the x and y axis respectively
    """
    Xmesh, Ymesh = np.meshgrid(CVdict_results['lam_values'], CVdict_results['d_tol_values'])
    r2_values  = np.array([i[-1] for i in CVdict_results['res_r2_score_history']]).reshape(len(CVdict_results['lam_values']),len(CVdict_results['d_tol_values']))
    plt.figure(figsize=(25, 16))
    plt.pcolormesh(Xmesh, Ymesh, r2_values.T, cmap='viridis')
    plt.colorbar(label='r2 values')
    plt.yticks(CVdict_results['d_tol_values'])
    plt.xticks(CVdict_results['lam_values'])
    plt.grid(True)
    plt.title('Your Plot Title')
    plt.xlabel('lambda')
    plt.ylabel('d_tol')
    plt.xticks(rotation=90)
    plt.show()

# def get_grid_vector_similarity(CVdict_results, distance_type="euclidean", colormap="hot"):
#     """
#     output the grid of r2 scores based on changing lambda and d_tol on the x and y axis respectively
#     """
#     Xmesh, Ymesh = np.meshgrid(CVdict_results['lam_values'], CVdict_results['d_tol_values'])
#     # r2_values  = np.array([i[-1] for i in CVdict_results['res_r2_score_history']]).reshape(len(CVdict_results['lam_values']),len(CVdict_results['d_tol_values']))
    
#     my_list = [i[-1] for i in CVdict_results['res_r2_score_history']]
#     best_r2_id = my_list.index(max(my_list))
#     best_res_vector = np.real(CVdict_results['res_c'][best_r2_id])

#     all_euclidean_distances_from_best_res = np.zeros((np.array(CVdict_results['res_c']).shape[0]))
#     for i in range(all_euclidean_distances_from_best_res.shape[0]):
#             all_euclidean_distances_from_best_res[i] = np.linalg.norm(best_res_vector - np.real(CVdict_results['res_c'][i]))
    
#     all_euclidean_distances_from_best_res.reshape(len(CVdict_results['lam_values']),len(CVdict_results['d_tol_values']))
    
#     plt.figure(figsize=(25, 16))
#     plt.pcolormesh(Xmesh, Ymesh, all_euclidean_distances_from_best_res.T, cmap=colormap)
#     plt.colorbar(label='r2 values')
#     plt.yticks(CVdict_results['d_tol_values'])
#     plt.xticks(CVdict_results['lam_values'])
#     plt.grid(True)
#     plt.title('Your Plot Title')
#     plt.xlabel('lambda')
#     plt.ylabel('d_tol')
#     plt.xticks(rotation=90)
#     plt.show()



def points_taken(data3darray,num_xy,num_t,boundary_x,start_img,skip_img):
    np.random.seed(0)
    m = data3darray.shape[0]
    n = data3darray.shape[1]
    points = {}
    count = 0
    for p in range(num_xy):
        x = np.random.choice(np.arange(boundary_x,n-boundary_x),1)[0]
        y = np.random.choice(np.arange(boundary_x,m-boundary_x),1)[0]
        for t in range(num_t):
            # points[count] = [x,y,2*t+12]
            # points[count] = [x, y, t] #2*t+12
            points[count] = [x, y, skip_img*t+start_img]
            count = count + 1
    print("time points taken")
    for t in range(num_t):
        print([skip_img*t+start_img], end = " ")
    print(" ")
    return points #dict of points
####
def sindy_makeX_desc_PolyDiffPoint(points,data_for_regres,boundary,boundary_x,num_points,degree, Fourier=False, deg=5,dt=1,dy=1,dx=1):
    """ 
    When using PolyDiffPoint, deg is the degree of the Chevishev polynomial to use for aprox the derivatives 
    degree: it is for the polynomials we built with the derivatives
    deg: it is for the degree of the chevisheb polynomial for the approximation of the derivatives
    """
    # Code Derivees
    ## Take up to second order derivatives.
    w = np.zeros((num_points,1))
    w2 = np.zeros((num_points,1))
    w3 = np.zeros((num_points,1))

    # u = np.zeros((num_points,1))
    # v = np.zeros((num_points,1))
    wt = np.zeros((num_points,1))
    wx = np.zeros((num_points,1))
    wxx = np.zeros((num_points,1))
    wxxx = np.zeros((num_points,1))
    wxxxx = np.zeros((num_points,1))

    wy = np.zeros((num_points,1))
    wyy = np.zeros((num_points,1))
    wyyy = np.zeros((num_points,1))
    wyyyy = np.zeros((num_points,1))


    wxy = np.zeros((num_points,1))
    wxyy = np.zeros((num_points,1))
    wxxy = np.zeros((num_points,1))
    wxyyy = np.zeros((num_points,1))
    wxxyy = np.zeros((num_points,1))
    wxxxy = np.zeros((num_points,1))

    
    N = 2*boundary-1  # odd number of points to use in fitting
    Nx = 2*boundary_x-1  # odd number of points to use in fitting
    deg = 5 # degree of polynomial to use
    print(f"N : {N}\nNx : {Nx}\ndeg : {deg}")
    for p in points.keys():    #<--- change points!
        #print(f"{p} _ {count}")
        [x, y, t] = points[p]  #<--- change points!
        w[p] = data_for_regres[x, y, t]
        w2[p] = w[p]**2
        w3[p] = w[p]**3

        # wx, wxx, wxxx, wxxxx, wy, wyy, wyyy, wyyyy, wxy, wxxy, wxxxy, wxxyy, wxyy, wxyyy
        wt[p] = PolyDiffPoint(data_for_regres[x, y, t-(N-1)//2:t+(N+1)//2], np.arange(N)*dt, deg, 1)[0]
        x_diff = PolyDiffPoint(data_for_regres[x-(Nx-1)//2:x+(Nx+1)//2, y, t], np.arange(Nx)*dx, deg, 4)
        y_diff = PolyDiffPoint(data_for_regres[x, y-(N-1)//2:y+(N+1)//2, t], np.arange(N)*dy, deg, 4)
        wx[p] = x_diff[0]
        wy[p] = y_diff[0]

        x_diff_yp = PolyDiffPoint(data_for_regres[x-(Nx-1)//2:x+(Nx+1)//2, y + 1, t], np.arange(Nx)*dx, deg, 3)
        x_diff_ym = PolyDiffPoint(data_for_regres[x-(Nx-1)//2:x+(Nx+1)//2, y - 1, t], np.arange(Nx)*dx, deg, 3)

        wxx[p] = x_diff[1]
        wxy[p] = (x_diff_yp[0]-x_diff_ym[0])/(2*dy)
        wyy[p] = y_diff[1]

        wxxx[p] = x_diff[2]
        wxxxx[p] = x_diff[3]

        wyyy[p] = y_diff[2]
        wyyyy[p] = y_diff[3]

        wxxy[p] = (x_diff_yp[1]-x_diff_ym[1])/(2*dy)
        wxxxy[p] = (x_diff_yp[2]-x_diff_ym[2])/(2*dy)

        y_diff_xm = PolyDiffPoint(data_for_regres[x+1, y-(Nx-1)//2:y+(Nx+1)//2, t], np.arange(Nx)*dy, deg, 3) #a droite
        y_diff_xp = PolyDiffPoint(data_for_regres[x+2, y-(Nx-1)//2:y+(Nx+1)//2, t], np.arange(Nx)*dy, deg, 3) #a gauche

        wxyy[p]  = (y_diff_xm[1]-y_diff_xp[1])/(2*dx)
        wxyyy[p] = (y_diff_xm[2]-y_diff_xp[2])/(2*dx)

        # wxxyy
        x_diff_ypp = PolyDiffPoint(data_for_regres[x-(Nx-1)//2:x+(Nx+1)//2, y + 2, t], np.arange(Nx)*dx, deg, 2)
        x_diff_ymm = PolyDiffPoint(data_for_regres[x-(Nx-1)//2:x+(Nx+1)//2, y - 2, t], np.arange(Nx)*dx, deg, 2)
        wxxyy[p] = (x_diff_ypp[1]-2*x_diff[1]+x_diff_ymm[1])/(4*dy)
    # Form Theta using up to quadratic polynomials in all variables.
    X_data = np.hstack([w])
    # X_ders = np.hstack([np.ones((num_points,1)), wx, wy, wxx, wxy, wyy])
    X_ders = np.hstack([ wx, wxx, wxxx, wxxxx, wy, wyy, wyyy, wyyyy, wxy, wxxy, wxxxy, wxxyy, wxyy, wxyyy])
    X_w = np.hstack([w,w2,w3])
    # X_ders_descr = ['','w_{x}', 'w_{y}','w_{xx}','w_{xy}','w_{yy}']
    X_ders_descr = ['', 'w_{x}', 'w_{xx}', 'w_{xxx}', 'w_{xxxx}', 'w_{y}', 'w_{yy}', 'w_{yyy}', 'w_{yyyy}','w_{xy}','w_{xxy}','w_{xxxy}','w_{xxyy}','w_{xyy}','w_{xyyy}']
    if Fourier==True:
        libpol = PolynomialLibrary(degree=degree)
        lib_fourier = FourierLibrary()
        lib_concat = ConcatLibrary([libpol,lib_fourier])
        lib_concat.fit(X_ders)
        X = lib_concat.transform(X_ders)
        lib_concat_featnames = lib_concat.get_feature_names()
        
        X = np.concatenate((X,X_w), axis=1)
        lib_concat_featnames.append("u")
        lib_concat_featnames.append("u^2")
        lib_concat_featnames.append("u^3")
        return X, lib_concat_featnames, wt
    else:
        libpol = PolynomialLibrary(degree=3)
        libpol.fit(X_ders)
        X = libpol.transform(X_ders)
        libpol_featnames = libpol.get_feature_names()
        return X, libpol_featnames, wt


def makeX_desc_PolyDiffPoint(points,data_for_regres,boundary,boundary_x,num_points, deg=5,dt=1,dy=1,dx=1):
    """ 
    When using PolyDiffPoint, deg is the degree of the Chevishev polynomial to use for aprox the derivatives
    """
    # Code Derivees
    ## Take up to second order derivatives.
    w = np.zeros((num_points,1))
    # u = np.zeros((num_points,1))
    # v = np.zeros((num_points,1))
    wt = np.zeros((num_points,1))
    wx = np.zeros((num_points,1))
    wxx = np.zeros((num_points,1))
    wxxx = np.zeros((num_points,1))
    wxxxx = np.zeros((num_points,1))

    wy = np.zeros((num_points,1))
    wyy = np.zeros((num_points,1))
    wyyy = np.zeros((num_points,1))
    wyyyy = np.zeros((num_points,1))


    wxy = np.zeros((num_points,1))
    wxyy = np.zeros((num_points,1))
    wxxy = np.zeros((num_points,1))
    wxyyy = np.zeros((num_points,1))
    wxxyy = np.zeros((num_points,1))
    wxxxy = np.zeros((num_points,1))

    N = 2*boundary-1  # odd number of points to use in fitting
    Nx = 2*boundary_x-1  # odd number of points to use in fitting
    deg = 5 # degree of polynomial to use
    print(f"N : {N}\nNx : {Nx}\ndeg : {deg}")
    for p in points.keys():    #<--- change points!
        #print(f"{p} _ {count}")
        [x, y, t] = points[p]  #<--- change points!
        w[p] = data_for_regres[x, y, t]

        # wx, wxx, wxxx, wxxxx, wy, wyy, wyyy, wyyyy, wxy, wxxy, wxxxy, wxxyy, wxyy, wxyyy
        wt[p] = PolyDiffPoint(data_for_regres[x, y, t-(N-1)//2:t+(N+1)//2], np.arange(N)*dt, deg, 1)[0]
        x_diff = PolyDiffPoint(data_for_regres[x-(Nx-1)//2:x+(Nx+1)//2, y, t], np.arange(Nx)*dx, deg, 4)
        y_diff = PolyDiffPoint(data_for_regres[x, y-(N-1)//2:y+(N+1)//2, t], np.arange(N)*dy, deg, 4)
        wx[p] = x_diff[0]
        wy[p] = y_diff[0]

        x_diff_yp = PolyDiffPoint(data_for_regres[x-(Nx-1)//2:x+(Nx+1)//2, y + 1, t], np.arange(Nx)*dx, deg, 3)
        x_diff_ym = PolyDiffPoint(data_for_regres[x-(Nx-1)//2:x+(Nx+1)//2, y - 1, t], np.arange(Nx)*dx, deg, 3)

        wxx[p] = x_diff[1]
        wxy[p] = (x_diff_yp[0]-x_diff_ym[0])/(2*dy)
        wyy[p] = y_diff[1]

        wxxx[p] = x_diff[2]
        wxxxx[p] = x_diff[3]

        wyyy[p] = y_diff[2]
        wyyyy[p] = y_diff[3]

        wxxy[p] = (x_diff_yp[1]-x_diff_ym[1])/(2*dy)
        wxxxy[p] = (x_diff_yp[2]-x_diff_ym[2])/(2*dy)

        y_diff_xm = PolyDiffPoint(data_for_regres[x+1, y-(Nx-1)//2:y+(Nx+1)//2, t], np.arange(Nx)*dy, deg, 3) #a droite
        y_diff_xp = PolyDiffPoint(data_for_regres[x+2, y-(Nx-1)//2:y+(Nx+1)//2, t], np.arange(Nx)*dy, deg, 3) #a gauche

        wxyy[p]  = (y_diff_xm[1]-y_diff_xp[1])/(2*dx)
        wxyyy[p] = (y_diff_xm[2]-y_diff_xp[2])/(2*dx)

        # wxxyy
        x_diff_ypp = PolyDiffPoint(data_for_regres[x-(Nx-1)//2:x+(Nx+1)//2, y + 2, t], np.arange(Nx)*dx, deg, 2)
        x_diff_ymm = PolyDiffPoint(data_for_regres[x-(Nx-1)//2:x+(Nx+1)//2, y - 2, t], np.arange(Nx)*dx, deg, 2)
        wxxyy[p] = (x_diff_ypp[1]-2*x_diff[1]+x_diff_ymm[1])/(4*dy)
    # Form Theta using up to quadratic polynomials in all variables.
    X_data = np.hstack([w])
    # X_ders = np.hstack([np.ones((num_points,1)), wx, wy, wxx, wxy, wyy])
    X_ders = np.hstack([np.ones((num_points, 1)), wx, wxx, wxxx, wxxxx, wy, wyy, wyyy, wyyyy, wxy, wxxy, wxxxy, wxxyy, wxyy, wxyyy])
    # X_ders_descr = ['','w_{x}', 'w_{y}','w_{xx}','w_{xy}','w_{yy}']
    X_ders_descr = ['', 'w_{x}', 'w_{xx}', 'w_{xxx}', 'w_{xxxx}', 'w_{y}', 'w_{yy}', 'w_{yyy}', 'w_{yyyy}','w_{xy}','w_{xxy}','w_{xxxy}','w_{xxyy}','w_{xyy}','w_{xyyy}']
    X, description = build_Theta(X_data, X_ders, X_ders_descr, 4, data_description = ['w']) #2
    print('Candidate terms for PDE')
    ['1']+description[1:]
    return X, description, wt
def CVtestTrainSTRidge(X, wt, lam_values, d_tol_values, offset_stop=0, loud=True):
    CVdict_results = {'res_c': [], 'res_best_tolerance':[],'res_train_error_history':[],'res_test_error_history':[],
                      'res_number_of_derivatives_history':[],'res_l0penalty_history':[],'res_r2_score_history':[],'res_mse_score_history':[],'res_mae_score_history':[],'res_rmae_score_history':[],
                      'res_TestY':[],'res_saved_TestR':[],'res_STRidge_err_history':[],
                      'res_STRidge_err_test_history':[],
                      'lam_values':lam_values,
                      'd_tol_values':d_tol_values}

    for i in tqdm(lam_values):
        for j in d_tol_values:
            l = 10**-6
            c, best_tolerance, train_error_history, test_error_history, number_of_derivatives_history, l0penalty_history, r2_score_history,mse_score_history,mae_score_history,rmae_score_history, TestY, saved_TestR, STRidge_err_history, STRidge_err_test_history = testTrainSTRidge(X, wt, lam=i, d_tol=j, offset_stop=offset_stop, loud=False)

            CVdict_results['res_c'].append(c)
            CVdict_results['res_best_tolerance'].append(best_tolerance)
            CVdict_results['res_train_error_history'].append(train_error_history)
            CVdict_results['res_test_error_history'].append(test_error_history)
            CVdict_results['res_number_of_derivatives_history'].append(number_of_derivatives_history)
            CVdict_results['res_l0penalty_history'].append(l0penalty_history)
            CVdict_results['res_r2_score_history'].append(r2_score_history)
            CVdict_results['res_mse_score_history'].append(mse_score_history)
            CVdict_results['res_mae_score_history'].append(mae_score_history)
            CVdict_results['res_rmae_score_history'].append(rmae_score_history)
            CVdict_results['res_TestY'].append(TestY)
            CVdict_results['res_saved_TestR'].append(saved_TestR)
            CVdict_results['res_STRidge_err_history'].append(STRidge_err_history)
            CVdict_results['res_STRidge_err_test_history'].append(STRidge_err_test_history)
    return CVdict_results

def input_to_regression(data3darray,num_xy,num_t,boundary,boundary_x,num_points,start_img,skip_img,degree=False, Fourier=False,sindy_lib=False):
    print("Starting point sampling.")
    points = points_taken(data3darray,num_xy,num_t,boundary_x,start_img,skip_img)
    print("Points sampling done.")
    print("Starting derivatives calculation.")
    if sindy_lib:
        X, description, wt = sindy_makeX_desc_PolyDiffPoint(points,data3darray,boundary,boundary_x,num_points,degree, Fourier, deg=5,dt=1,dy=1,dx=1)
        print("All done.")
        return X, description, wt, points
    X, description, wt = makeX_desc_PolyDiffPoint(points,data3darray,boundary,boundary_x,num_points, deg=5,dt=1,dy=1,dx=1)
    print("All done.")
    return X, description, wt, points
    


def calc_images_n_entropy(list_of_images3Darrays):
    #calculate sequency of entropies for each element of list_of_images3Darrays 
    list_of_entropies =  [[] for _ in range(len(list_of_images3Darrays))]
    #print(list_of_entropies)
    for i in range(len(list_of_images3Darrays)):
        for j in list_of_images3Darrays[i]:
            #print(i)
            list_of_entropies[i].append(entropy(j.flatten()))
            #print(j.shape)
            #list_of_entropies[i].append(entropy(i[j,:,:].flatten()))
    return np.asarray(list_of_entropies)

def get_centered_patches(data_in_3d_array, size=200):
    center_y = data_in_3d_array[0].shape[0] // 2
    center_x = data_in_3d_array[0].shape[1] // 2
    halfsize= size//2
    data_patches = data_in_3d_array[:,center_y-(halfsize):center_y+(halfsize),
                                    center_x-(halfsize):center_x+(halfsize)]
    return np.asarray(data_patches)
def plot_entropies(list_2d_arrays, list_col_titles=None, frequency=1):
    # Each list must have 2d arrays with the same number in dimension 0
    cols = len(list_2d_arrays)
    rows = list_2d_arrays[0].shape[0]
    fig, ax = plt.subplots(rows,cols,sharex=True) #, sharex=True,sharey=True)
    for r in range(rows):
        for c in range(cols):
            ax[r,c].plot(list_2d_arrays[c][r,::frequency])
            if list_col_titles and r==0:
                ax[r,c].title.set_text(list_col_titles[c])
    default_x_ticks=range(list_2d_arrays[0][0,::frequency].shape[0])
    #plt.xticks(np.array([0]),np.array([-1]))
    #plt.xticks(default_x_ticks, default_x_ticks)
    plt.xticks([default_x_ticks[0], default_x_ticks[-1]], labels=['Start', 'End'])
    plt.show()


def get_misal_from_loaded(list_of_SH_datasets_misal, nro_of_mis=5):
    # get a higher number of nro_of_mis might crash the python kernel
    dim0 = len(list_of_SH_datasets_misal)
    dim1 = nro_of_mis
    dim2 = list_of_SH_datasets_misal[0][0][0].shape[0]
    dim3 = list_of_SH_datasets_misal[0][0][0].shape[1]
    misarrays = np.empty((dim0,dim1,dim2,dim3))
    for i in range(dim0):
        for j in range(dim1):
            misarrays[i,j,:,:] = list_of_SH_datasets_misal[i][j][0]
    return misarrays


def calc_imagesmisaligned_n_entropy(missalignments_4darray):
    #calculate sequency of entropies for each element of list_of_images3Darrays 
    array_entropies =  np.empty((missalignments_4darray.shape[0],missalignments_4darray.shape[1],missalignments_4darray.shape[2]))
    #print(list_of_entropies)
    for i in range(missalignments_4darray.shape[0]):
        for j in range(missalignments_4darray.shape[1]):
            for k in range(missalignments_4darray.shape[2]):
                array_entropies[i,j,k] = entropy(missalignments_4darray[i,j,k,:])
    return array_entropies


def plot_miss_ent_estimated_ent(data,ent_data,freq,frac_lowess):
    rows = data.shape[1]
    cols = data.shape[0]
    fig, axs = plt.subplots(rows,cols, sharex=True)
    for misal_rnd in range(rows):
        for sh_exp in range(cols):
            Y_var_for_ent = data[sh_exp,misal_rnd,::freq].T
            #Y_var_for_ent = ent_misalingments60[experimt,:][:,::8].T
            X_var_for_ent = np.arange(0,Y_var_for_ent.shape[0])*freq
            lowess = sm.nonparametric.lowess(Y_var_for_ent, X_var_for_ent, frac=frac_lowess)
            axs[misal_rnd,sh_exp].scatter(X_var_for_ent.reshape(-1,1),Y_var_for_ent, s=0.1)
            axs[misal_rnd,sh_exp].plot(lowess[:, 0], lowess[:, 1], color='red', label='LOWESS Smoothing')
            axs[misal_rnd,sh_exp].plot(ent_data[sh_exp,:], color='k')
            my_pwlf = pwlf.PiecewiseLinFit(X_var_for_ent, Y_var_for_ent)
            my_pwlf.fit(3)
            y_pwlf_pred = my_pwlf.predict(X_var_for_ent)
            axs[misal_rnd,sh_exp].plot(X_var_for_ent,y_pwlf_pred, color='seagreen')
    default_x_ticks=range(X_var_for_ent.shape[0])
    plt.xticks([default_x_ticks[0], default_x_ticks[-1]], labels=['Start', 'End'])
    plt.show()


def calc_and_plot_lowess_for_lab_data_ent(ent_1darray, start, end, color, frac=0.3):
    x_val_lowess = np.arange(start,end)
    x_val_ent = np.arange(len(ent_1darray))
    lowess_result = sm.nonparametric.lowess(ent_1darray[start:end], x_val_lowess, frac=frac)
    plt.plot(x_val_ent,ent_1darray)
    plt.plot(lowess_result[:, 0], lowess_result[:, 1], color=color, label='LOWESS')
    plt.show()
    return x_val_lowess, lowess_result



"""
A few functions used in PDE-FIND

Samuel Rudy.  2016

"""
def huberRidge(X0, y, lam, maxit, tol, epsilon=1.35, normalize = 2, print_results = False, offset_stop=0, loud=False, l0_penalty = None, X0test=None, ytest=None):
    """

    """
    outliers_history = []
    n, d = X0.shape
    X = np.zeros((n,d), dtype=np.float32)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:, i] ,normalize))
            X[:,i] = Mreg[i] * X0[:, i]
    else:
        X = X0
    
    #ntest, dtest = X0test.shape
    #Xtest = np.zeros((ntest,dtest), dtype=np.float32)
    ## First normalize data
    #if normalize != 0:
    #    Mregtest = np.zeros((dtest, 1))
    #    for i in range(0,dtest):
    #        Mregtest[i] = 1.0/(np.linalg.norm(X0test[:, i] ,normalize))
    #        Xtest[:,i] = Mregtest[i] * X0test[:, i]
    #else:
    #    Xtest = X0test

    # Get the standard ridge esitmate
    if lam != 0:
        huber_first = HuberRegressor(alpha=lam,epsilon=epsilon).fit(X, y)
        w = huber_first.coef_
        print("huber_first ",len(w), type(w))
        #w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y),rcond=None)[0]
    else:
        huber_alpha_0 = HuberRegressor(alpha=lam,epsilon=epsilon).fit(X, y)
        w_alpha_0 = huber_alpha_0.coef_
        #w = np.linalg.lstsq(X, y, rcond=None)[0]
    num_relevant = d
    biginds = np.where(abs(w) >= tol)[0]
    print("len biginds:", len(biginds))
    if loud:
        # print(f"Get Standard Ridge Estimate : {w}")
        print(f"number of relevant elements : {num_relevant}\nThe relevant elements : {biginds}")
    
    #err_history = []
    #err_test_history = []
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
        print(f'iteration {j}; len new_biginds {len(new_biginds)}')
        if loud:
            print(f"Items to cut out because they're smaller than the tolerance : {smallinds}\nnew relevant elements : {new_biginds}")
        
        # Get error, r2_score, number of remaining derivatives
        
        #err = np.linalg.norm(y - X.dot(w), 2) + l0_penalty*np.count_nonzero(w)
        #err_history.append(err)
        #err_test = np.linalg.norm(ytest - Xtest.dot(w), 2) + l0_penalty*np.count_nonzero(w)
        #err_test_history.append(err_test)

        # If nothing changes then stop
        if num_relevant == len(new_biginds) and j >= offset_stop:
            if loud:
                print(f"Nothing has changed after iteration number {j} out of {maxit}. We are stopping.")
            break
        else:
            num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                # if print_results: print "Tolerance too high - all coefficients set below tolerance"
                if loud:
                    print("TOLERANCE TOO HIGH! Tolerance too high - all coefficients set below tolerance.")
                return w #, err_history, err_test_history
            else:
                break
        biginds = new_biginds
        
        # Otherwise get a new guess
        if loud:
            print(f"We get another guess. Another ridge regression is up next. We are at iteration {j} out of {maxit}")
        w[smallinds] = 0
        if lam != 0:
            huber_second = HuberRegressor(alpha=lam,epsilon=epsilon).fit(X[:, biginds], y)
            w_second = huber_second.coef_
            huber_second_outliers = huber_second.outliers_
            outliers_history.append(huber_second_outliers)
            for ind,val in enumerate(biginds):
                w[val]=w_second[ind]
            #w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=None)[0]
        else:
            print("WARNING we shouldnt be here")
            w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]


    # Now that we have the sparsity pattern, use standard least squares to get w
    #if len(biginds) != 0: 

    #    w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]
    # print(f"err history : {err_history}")
    if normalize != 0:
        print("Mreg.shape",Mreg.shape)
        print("w.shape",w.shape)
        final_w = np.multiply(Mreg, w.reshape(-1,1))
        return final_w, outliers_history#, err_history, err_test_history
    else:
        return w, outliers_history#, err_history, err_test_history

def testSTRidge(X0, y, lam, maxit, tol, normalize = 2, print_results = False, offset_stop=0, loud=False, l0_penalty = None, X0test=None, ytest=None):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n, d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:, i] ,normalize))
            X[:,i] = Mreg[i] * X0[:, i]
    else:
        X = X0
    
    ntest, dtest = X0test.shape
    Xtest = np.zeros((ntest,dtest), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mregtest = np.zeros((dtest, 1))
        for i in range(0,dtest):
            Mregtest[i] = 1.0/(np.linalg.norm(X0test[:, i] ,normalize))
            Xtest[:,i] = Mregtest[i] * X0test[:, i]
    else:
        Xtest = X0test

    # Get the standard ridge esitmate
    if lam != 0:
        w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y),rcond=None)[0]
    else:
        w = np.linalg.lstsq(X, y, rcond=None)[0]
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]
    
    if loud:
        # print(f"Get Standard Ridge Estimate : {w}")
        print(f"number of relevant elements : {num_relevant}\nThe relevant elements : {biginds}")
    
    err_history = []
    err_test_history = []
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
        if loud:
            print(f"Items to cut out because they're smaller than the tolerance : {smallinds}\nnew relevant elements : {new_biginds}")
        
        # Get error, r2_score, number of remaining derivatives
        err = np.linalg.norm(y - X.dot(w), 2) + l0_penalty*np.count_nonzero(w)
        err_history.append(err)
        err_test = np.linalg.norm(ytest - Xtest.dot(w), 2) + l0_penalty*np.count_nonzero(w)
        err_test_history.append(err_test)

        # If nothing changes then stop
        if num_relevant == len(new_biginds) and j >= offset_stop:
            if loud:
                print(f"Nothing has changed after iteration number {j} out of {maxit}. We are stopping.")
            break
        else:
            num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                # if print_results: print "Tolerance too high - all coefficients set below tolerance"
                if loud:
                    print("Tolerance too high - all coefficients set below tolerance.")
                return w, err_history, err_test_history
            else:
                break
        biginds = new_biginds
        
        # Otherwise get a new guess
        if loud:
            print(f"We get another guess. Another ridge regression is up next. We are at iteration {j} out of {maxit}")
        w[smallinds] = 0
        if lam != 0:
            w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=None)[0]
        else:
            w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]


    # Now that we have the sparsity pattern, use standard least squares to get w
    if len(biginds) != 0: 
        w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]
    # print(f"err history : {err_history}")
    if normalize != 0:
        return np.multiply(Mreg, w), err_history, err_test_history
    else:
        return w, err_history, err_test_history

def testTrainSTRidge(R, Ut, lam, d_tol, maxit = 25, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, print_best_tol = False, offset_stop=0, loud=False):
    """
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a training set, then evaluates them 
    using a loss function on a holdout set.

    Please note published article has typo.  Loss function used here for model selection evaluates fidelity using 2-norm,
    not squared 2-norm.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train, :]
    TestR = R[test, :]
    TrainY = Ut[train, :]
    TestY = Ut[test, :]
    D = TrainR.shape[1]

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None:
        l0_penalty = 0.001 * np.linalg.cond(R)
    if loud:
        print(f"l0_penalty : {np.linalg.cond(R)}")

    if loud:
        print(f"Initial tolerance : {d_tol}, Initial l0_penalty : {l0_penalty}")
    # Get the standard least squares estimator
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY, rcond=None)[0]
    train_err = np.linalg.norm(TrainY - TrainR.dot(w), 2) + l0_penalty*np.count_nonzero(w_best)
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty * np.count_nonzero(w_best)
    
    # l1 = []
    # l2 = []
    # print(TestR.dot(w_best)[0], type(TestR.dot(w_best)[0]), list(TestR.dot(w_best)[0]), np.real(TestR.dot(w_best)[0]))
    # _ = [l1.append(np.real(x[0])) for x in TestY]
    # _ = [l2.append(np.real(x[0])) for x in TestR.dot(w_best)]
    # r2score = r2_score(l1, l2)
    r2score = r2_score(TestY, np.real(TestR.dot(w_best)))
    mse_score = mean_squared_error(TestY, np.real(TestR.dot(w_best)))
    mae_score = mean_absolute_error(TestY, np.real(TestR.dot(w_best)))
    rmae=mae_score/(np.max(TestY)-np.min(TestY))

    tol_best = 0
    if loud:
        print(f"Initial Least Squares :\nw_best : {w_best}\nerr_best : {err_best}\ntol_best : {tol_best}")
        print("Now we start the main loop, increasing the tolerance until test performance decreases")
    
    train_err_history = [train_err]
    test_err_history = [err_best]
    number_of_derivatives_history = [np.count_nonzero(w_best)]
    l0penalty_history = [l0_penalty * np.count_nonzero(w_best)]
    r2_score_history = [r2score]
    mse_score_history = [mse_score]
    mae_score_history = [mae_score]
    rmae_score_history =[rmae]
    STRidge_err_history = []
    STRidge_err_test_history = []

    saved_TestR = TestR.dot(w_best)
    # [print(x) for x in zip(l1[:10], l2[:10])]
    # Now increase tolerance until test performance decreases
    for iter in range(maxit):
        if loud:
            print(f"\nTrainSTRidge iteration number {iter} out of {maxit}\n")
        # Get a set of coefficients and error
        w, STRidge_err, STRidge_err_test = testSTRidge(TrainR, TrainY, lam, STR_iters, tol, normalize=normalize, offset_stop=offset_stop, loud=loud, l0_penalty=l0_penalty, X0test=TestR, ytest=TestY)
        train_err = np.linalg.norm(TrainY - TrainR.dot(w), 2) + l0_penalty*np.count_nonzero(w)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty*np.count_nonzero(w)
        # l1 = []
        # l2 = []
        # _ = [l1.append(np.real(x[0])) for x in TestY]
        # _ = [l2.append(np.real(x[0])) for x in TestR.dot(w_best)]
        # r2score = r2_score(l1, l2)
        r2score = r2_score(TestY, np.real(TestR.dot(w_best)))
        mse_score = mean_squared_error(TestY, np.real(TestR.dot(w_best)))
        mae_score = mean_absolute_error(TestY, np.real(TestR.dot(w_best)))
        rmae=mae_score/(np.max(TestY)-np.min(TestY))
        train_err_history.append(train_err)
        test_err_history.append(err)
        number_of_derivatives_history.append(np.count_nonzero(w))
        l0penalty_history.append(l0_penalty*np.count_nonzero(w))
        r2_score_history.append(r2score)
        mse_score_history.append(mse_score)
        mae_score_history.append(mae_score)
        rmae_score_history.append(rmae)
        STRidge_err_zeros = np.zeros(10 - len(STRidge_err))
        STRidge_err_history.extend(STRidge_err)
        STRidge_err_history.extend(STRidge_err_zeros)
        STRidge_err_zeros = np.zeros(10 - len(STRidge_err_test))
        STRidge_err_test_history.extend(STRidge_err_test)
        STRidge_err_test_history.extend(STRidge_err_zeros)
        if loud:
            print(f"Calculating error : {np.linalg.norm(TestY - TestR.dot(w), 2)} + {l0_penalty}*{np.count_nonzero(w)}")
            print(f"current best error : {err_best}, new error : {err}")
        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0, tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol:
        print("Optimal tolerance:", tol_best)

    return w_best, tol_best, train_err_history, test_err_history, number_of_derivatives_history, l0penalty_history, r2_score_history,mse_score_history,mae_score_history,rmae_score_history, TestY, saved_TestR, STRidge_err_history, STRidge_err_test_history


##################################################################################
##################################################################################
#
# Functions for taking derivatives.
# When in doubt / nice data ===> finite differences
#               \ noisy data ===> polynomials
#             
##################################################################################
##################################################################################

def TikhonovDiff(f, dx, lam, d = 1):
    """
    Tikhonov differentiation.

    return argmin_g \|Ag-f\|_2^2 + lam*\|Dg\|_2^2
    where A is trapezoidal integration and D is finite differences for first dervative

    It looks like it will work well and does for the ODE case but 
    tends to introduce too much bias to work well for PDEs.  If the data is noisy, try using
    polynomials instead.
    """

    # Initialize a few things    
    n = len(f)
    f = np.matrix(f - f[0]).reshape((n,1))

    # Get a trapezoidal approximation to an integral
    A = np.zeros((n,n))
    for i in range(1, n):
        A[i,i] = dx/2
        A[i,0] = dx/2
        for j in range(1,i): A[i,j] = dx
    
    e = np.ones(n-1)
    D = sparse.diags([e, -e], [1, 0], shape=(n-1, n)).todense() / dx
    
    # Invert to find derivative
    g = np.squeeze(np.asarray(np.linalg.lstsq(A.T.dot(A) + lam*D.T.dot(D),A.T.dot(f),rcond=None)[0]))
    
    if d == 1: return g

    # If looking for a higher order derivative, this one should be smooth so now we can use finite differences
    else: return FiniteDiff(g, dx, d-1)
    
def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3
    
    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """
    
    n = u.size
    ux = np.zeros(n, dtype=np.complex64)
    
    if d == 1:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-u[i-1]) / (2*dx)
        
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux
    
    if d == 2:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2
        
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux
    
    if d == 3:
        for i in range(2,n-2):
            ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3
        
        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux
    
    if d > 3:
        return FiniteDiff(FiniteDiff(u,dx,3), dx, d-3)
    
def ConvSmoother(x, p, sigma):
    """
    Smoother for noisy data
    
    Inpute = x, p, sigma
    x = one dimensional series to be smoothed
    p = width of smoother
    sigma = standard deviation of gaussian smoothing kernel
    """
    
    n = len(x)
    y = np.zeros(n, dtype=np.complex64)
    g = np.exp(-np.power(np.linspace(-p,p,2*p),2)/(2.0*sigma**2))

    for i in range(n):
        a = max([i-p,0])
        b = min([i+p,n])
        c = max([0, p-i])
        d = min([2*p,p+n-i])
        y[i] = np.sum(np.multiply(x[a:b], g[c:d]))/np.sum(g[c:d])
        
    return y

def PolyDiff(u, x, deg = 3, diff = 1, width = 5):
    
    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2*width,diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        # Note code originally used an even number of points here.
        # This is an oversight in the original code fixed in 2022.
        points = np.arange(j - width, j + width + 1)

        # Fit to a polynomial
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives
        for d in range(1,diff+1):
            du[j-width, d-1] = poly.deriv(m=d)(x[j])

    return du

def PolyDiffPoint(u, x, deg = 3, diff = 1, index = None):
    
    """
    Same as above but now just looking at a single point

    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    """
    
    n = len(x)
    if index == None: index = (n-1)//2

    # Fit to a polynomial
    poly = np.polynomial.chebyshev.Chebyshev.fit(x,u,deg)
    
    # Take derivatives
    derivatives = []
    for d in range(1,diff+1):
        derivatives.append(poly.deriv(m=d)(x[index]))
        
    return derivatives

##################################################################################
##################################################################################
#
# Functions specific to PDE-FIND
#               
##################################################################################
##################################################################################

def build_Theta(data, derivatives, derivatives_description, P, data_description = None):
    """
    builds a matrix with columns representing polynoimials up to degree P of all variables

    This is used when we subsample and take all the derivatives point by point or if there is an 
    extra input (Q in the paper) to put in.

    input:
        data: column 0 is U, and columns 1:end are Q
        derivatives: a bunch of derivatives of U and maybe Q, should start with a column of ones
        derivatives_description: description of what derivatives have been passed in
        P: max power of polynomial function of U to be included in Theta

    returns:
        Theta = Theta(U,Q)
        descr = description of what all the columns in Theta are
    """
    
    n,d = data.shape
    m, d2 = derivatives.shape
    if n != m: raise Exception('dimension error')
    if data_description is not None: 
        if len(data_description) != d: raise Exception('data descrption error')
    
    # Create a list of all polynomials in d variables up to degree P
    rhs_functions = {}
    f = lambda x, y : np.prod(np.power(list(x), list(y)))
    powers = []            
    for p in range(1,P+1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d-1):
                starts = [0] + [index+1 for index in indices]
                stops = indices + (size,)
                powers.append(tuple(map(operator.sub, stops, starts)))
    for power in powers: rhs_functions[power] = [lambda x, y = power: f(x,y), power]

    # First column of Theta is just ones.
    Theta = np.ones((n,1), dtype=np.complex64)
    descr = ['']
    
    # Add the derivaitves onto Theta
    for D in range(1,derivatives.shape[1]):
        Theta = np.hstack([Theta, derivatives[:,D].reshape(n,1)])
        descr.append(derivatives_description[D])
        
    # Add on derivatives times polynomials
    for D in range(derivatives.shape[1]):
        for k in rhs_functions.keys():
            func = rhs_functions[k][0]
            new_column = np.zeros((n,1), dtype=np.complex64)
            for i in range(n):
                new_column[i] = func(data[i,:])*derivatives[i,D]
            Theta = np.hstack([Theta, new_column])
            if data_description is None: descr.append(str(rhs_functions[k][1]) + derivatives_description[D])
            else:
                function_description = ''
                for j in range(d):
                    if rhs_functions[k][1][j] != 0:
                        if rhs_functions[k][1][j] == 1:
                            function_description = function_description + data_description[j]
                        else:
                            function_description = function_description + data_description[j] + '^' + str(rhs_functions[k][1][j])
                descr.append(function_description + derivatives_description[D])

    return Theta, descr

def build_linear_system(u, dt, dx, D = 3, P = 3,time_diff = 'poly',space_diff = 'poly',lam_t = None,lam_x = None, width_x = None,width_t = None, deg_x = 5,deg_t = None,sigma = 2):
    """
    Constructs a large linear system to use in later regression for finding PDE.  
    This function works when we are not subsampling the data or adding in any forcing.

    Input:
        Required:
            u = data to be fit to a pde
            dt = temporal grid spacing
            dx = spatial grid spacing
        Optional:
            D = max derivative to include in rhs (default = 3)
            P = max power of u to include in rhs (default = 3)
            time_diff = method for taking time derivative
                        options = 'poly', 'FD', 'FDconv','TV'
                        'poly' (default) = interpolation with polynomial 
                        'FD' = standard finite differences
                        'FDconv' = finite differences with convolutional smoothing 
                                   before and after along x-axis at each timestep
                        'Tik' = Tikhonov (takes very long time)
            space_diff = same as time_diff with added option, 'Fourier' = differentiation via FFT
            lam_t = penalization for L2 norm of second time derivative
                    only applies if time_diff = 'TV'
                    default = 1.0/(number of timesteps)
            lam_x = penalization for L2 norm of (n+1)st spatial derivative
                    default = 1.0/(number of gridpoints)
            width_x = number of points to use in polynomial interpolation for x derivatives
                      or width of convolutional smoother in x direction if using FDconv
            width_t = number of points to use in polynomial interpolation for t derivatives
            deg_x = degree of polynomial to differentiate x
            deg_t = degree of polynomial to differentiate t
            sigma = standard deviation of gaussian smoother
                    only applies if time_diff = 'FDconv'
                    default = 2
    Output:
        ut = column vector of length u.size
        R = matrix with ((D+1)*(P+1)) of column, each as large as ut
        rhs_description = description of what each column in R is
    """

    n, m = u.shape

    if width_x == None: width_x = n//10
    if width_t == None: width_t = m//10
    if deg_t == None: deg_t = deg_x

    # If we're using polynomials to take derviatives, then we toss the data around the edges.
    if time_diff == 'poly': 
        m2 = m-2*width_t
        offset_t = width_t
    else: 
        m2 = m
        offset_t = 0
    if space_diff == 'poly': 
        n2 = n-2*width_x
        offset_x = width_x
    else: 
        n2 = n
        offset_x = 0

    if lam_t == None: lam_t = 1.0/m
    if lam_x == None: lam_x = 1.0/n

    ########################
    # First take the time derivaitve for the left hand side of the equation
    ########################
    ut = np.zeros((n2,m2), dtype=np.complex64)

    if time_diff == 'FDconv':
        Usmooth = np.zeros((n,m), dtype=np.complex64)
        # Smooth across x cross-sections
        for j in range(m):
            Usmooth[:,j] = ConvSmoother(u[:,j],width_t,sigma)
        # Now take finite differences
        for i in range(n2):
            ut[i,:] = FiniteDiff(Usmooth[i + offset_x,:],dt,1)

    elif time_diff == 'poly':
        T= np.linspace(0,(m-1)*dt,m)
        for i in range(n2):
            ut[i,:] = PolyDiff(u[i+offset_x,:],T,diff=1,width=width_t,deg=deg_t)[:,0]

    elif time_diff == 'Tik':
        for i in range(n2):
            ut[i,:] = TikhonovDiff(u[i + offset_x,:], dt, lam_t)

    else:
        for i in range(n2):
            ut[i,:] = FiniteDiff(u[i + offset_x,:],dt,1)
    
    ut = np.reshape(ut, (n2*m2,1), order='F')

    ########################
    # Now form the rhs one column at a time, and record what each one is
    ########################

    u2 = u[offset_x:n-offset_x,offset_t:m-offset_t]
    Theta = np.zeros((n2*m2, (D+1)*(P+1)), dtype=np.complex64)
    ux = np.zeros((n2,m2), dtype=np.complex64)
    rhs_description = ['' for i in range((D+1)*(P+1))]

    if space_diff == 'poly': 
        Du = {}
        for i in range(m2):
            Du[i] = PolyDiff(u[:,i+offset_t],np.linspace(0,(n-1)*dx,n),diff=D,width=width_x,deg=deg_x)
    if space_diff == 'Fourier': ik = 1j*np.fft.fftfreq(n)*n
        
    for d in range(D+1):

        if d > 0:
            for i in range(m2):
                if space_diff == 'Tik': ux[:,i] = TikhonovDiff(u[:,i+offset_t], dx, lam_x, d=d)
                elif space_diff == 'FDconv':
                    Usmooth = ConvSmoother(u[:,i+offset_t],width_x,sigma)
                    ux[:,i] = FiniteDiff(Usmooth,dx,d)
                elif space_diff == 'FD': ux[:,i] = FiniteDiff(u[:,i+offset_t],dx,d)
                elif space_diff == 'poly': ux[:,i] = Du[i][:,d-1]
                elif space_diff == 'Fourier': ux[:,i] = np.fft.ifft(ik**d*np.fft.fft(ux[:,i]))
        else: ux = np.ones((n2,m2), dtype=np.complex64) 
            
        for p in range(P+1):
            Theta[:, d*(P+1)+p] = np.reshape(np.multiply(ux, np.power(u2,p)), (n2*m2), order='F')

            if p == 1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u'
            elif p>1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u^' + str(p)
            if d > 0: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+\
                                                   'u_{' + ''.join(['x' for _ in range(d)]) + '}'

    return ut, Theta, rhs_description

def print_pde(w, rhs_description, ut = 'u_t', print_imag=False):
    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i].real != 0:
            if not first:
                pde = pde + ' + '
            if print_imag:
                pde = pde + "(%05f %+05fi)" % (w[i].real, w[i].imag) + rhs_description[i] + "\n   "
            else:
                pde = pde + "(%05f)" % (w[i].real) + rhs_description[i] + "\n   "
            first = False
    print(pde)

def print_coef_sindy_or_not(c,description):
    if np.sum(np.real(c)!=0)==0:
        return 0
    """
    Print coeficients, the derivatives original description is hard coded
    """
    desc_copy = copy.deepcopy(description)
    original_desc = ['', 'w_{x}', 'w_{xx}', 'w_{xxx}', 'w_{xxxx}', 'w_{y}', 'w_{yy}', 'w_{yyy}', 'w_{yyyy}','w_{xy}','w_{xxy}','w_{xxxy}','w_{xxyy}','w_{xyy}','w_{xyyy}']
    # Define the replacements (e.g., replace 'a' with 'x' and 'e' with 'y')
    replacements = {'x13': original_desc[-1], 'x12': original_desc[-2],'x11': original_desc[-3],'x10': original_desc[-4],'x9': original_desc[-5], 'x8': original_desc[-6],'x7': original_desc[-7],'x6': original_desc[-8],'x5': original_desc[-9],'x4': original_desc[-10], 'x3': original_desc[-11],'x2': original_desc[-12],'x1': original_desc[-13],'x0': original_desc[-14] }
    # Replace occurrences of specified letters in each string
    for i in range(len(desc_copy)):
        for j in replacements.keys():
            if j in desc_copy[i]:
                #print(desc_copy[i],j,replacements[j])
                desc_copy[i]=desc_copy[i].replace(j, replacements[j])
                #print("desc_copy[i]", desc_copy[i])

    if c.shape[0] != len(desc_copy):
        print("warning: desc length and c size not the same")
    c_strings = [str(np.real(i)) for i in c]
    polynomial=[]
    conditions = (np.real(c)!=0)
    for i in range(len(desc_copy)):
        polynomial.append(c_strings[i] + "*" + desc_copy[i])
    selected_values = [value for value, condition in zip(polynomial, conditions) if condition]
    print("w_t = ", end=" ")
    for i in selected_values[:-1]:
        print(f"{i} +", end=" ")
    print(selected_values[-1])
    ## selected_values= ["w_t = "+ i for i in selected_values]
    #return selected_values

        
##################################################################################
##################################################################################
#
# Functions for sparse regression.
#               
##################################################################################
##################################################################################

def TrainSTRidge(R, Ut, lam, d_tol, maxit = 25, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, print_best_tol = False):
    """
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a training set, then evaluates them 
    using a loss function on a holdout set.

    Please note published article has typo.  Loss function used here for model selection evaluates fidelity using 2-norm,
    not squared 2-norm.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train,:]
    TestR = R[test,:]
    TrainY = Ut[train,:]
    TestY = Ut[test,:]
    D = TrainR.shape[1]       

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None: 
        l0_penalty = 0.001*np.linalg.cond(R)
        print(l0_penalty)

    # Get the standard least squares estimator
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY,rcond=None)[0]
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty*np.count_nonzero(w_best)
    tol_best = 0

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        #print("hola")
        w = STRidge(TrainR,TrainY,lam,STR_iters,tol,normalize = normalize)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty*np.count_nonzero(w)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0,tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: print("Optimal tolerance:", tol_best)

    return w_best, err, tol_best

def Lasso(X0, Y, lam, w = np.array([0]), maxit = 100, normalize = 2):
    """
    Uses accelerated proximal gradient (FISTA) to solve Lasso
    argmin (1/2)*||Xw-Y||_2^2 + lam||w||_1
    """
    
    # Obtain size of X
    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    Y = Y.reshape(n,1)
    
    # Create w if none is given
    if w.size != d:
        w = np.zeros((d,1), dtype=np.complex64)
    w_old = np.zeros((d,1), dtype=np.complex64)
        
    # Initialize a few other parameters
    converge = 0
    objective = np.zeros((maxit,1))
    
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0

    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(X.T.dot(X),2)
    
    # Now loop until converged or max iterations
    for iters in range(0, maxit):
         
        # Update w
        z = w + iters/float(iters+1)*(w - w_old)
        w_old = w
        z = z - X.T.dot(X.dot(z)-Y)/L
        for j in range(d): w[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j])-lam/L,0]))

        # Could put in some sort of break condition based on convergence here.
    
    # Now that we have the sparsity pattern, used least squares.
    biginds = np.where(w != 0)[0]
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],Y,rcond=None)[0]

    # Finally, reverse the regularization so as to be able to use with raw data
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w

def ElasticNet(X0, Y, lam1, lam2, w = np.array([0]), maxit = 100, normalize = 2):
    """
    Uses accelerated proximal gradient (FISTA) to solve elastic net
    argmin (1/2)*||Xw-Y||_2^2 + lam_1||w||_1 + (1/2)*lam_2||w||_2^2
    """
    
    # Obtain size of X
    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    Y = Y.reshape(n,1)
    
    # Create w if none is given
    if w.size != d:
        w = np.zeros((d,1), dtype=np.complex64)
    w_old = np.zeros((d,1), dtype=np.complex64)
        
    # Initialize a few other parameters
    converge = 0
    objective = np.zeros((maxit,1))
    
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0

    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(X.T.dot(X),2) + lam2
    
    # Now loop until converged or max iterations
    for iters in range(0, maxit):
         
        # Update w
        z = w + iters/float(iters+1)*(w - w_old)
        w_old = w
        z = z - (lam2*z + X.T.dot(X.dot(z)-Y))/L
        for j in range(d): w[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j])-lam1/L,0]))

        # Could put in some sort of break condition based on convergence here.
    
    # Now that we have the sparsity pattern, used least squares.
    biginds = np.where(w != 0)[0]
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],Y,rcond=None)[0]

    # Finally, reverse the regularization so as to be able to use with raw data
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w
    
def STRidge(X0, y, lam, maxit, tol, normalize = 2, print_results = False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    
    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y),rcond=None)[0]
    else: w = np.linalg.lstsq(X,y,rcond=None)[0]
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]
    #print("w", w)
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else: break
        biginds = new_biginds
        
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=None)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    #print(type(np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]))
    #print(np.linalg.lstsq(X[:, biginds],y,rcond=None)[0].shape)

    #print(w.shape, biginds)
    if list(biginds) != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]
    
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w
    
def FoBaGreedy(X, y, epsilon = 0.1, maxit_f = 100, maxit_b = 5, backwards_freq = 5, relearn_f=True, relearn_b=True):
    """
    Forward-Backward greedy algorithm for sparse regression.
    See Zhang, Tom. 'Adaptive Forward-Backward Greedy Algorithm for Sparse Learning with Linear Models', NIPS, 2008

    The original version of this code that was uploaded github was contained errors.  This version has been corrected and
    also includes an variation of FoBa used in Thaler et al. 'Sparse identification of truncation errors,' JCP, 2019,where 
    we have additionally used relearning on the backwards step.  This later implementation (currently set as the default 
    with relearn_f=relearn_b=True) relearns non-zero terms of w, rather than only fitting the residual, as was done in Zhang.  
    It is slower, more robust, but still in some cases underperforms STRidge.
    """

    n,d = X.shape
    F = {}
    F[0] = set()
    w = {}
    w[0] = np.zeros((d,1))
    k = 0
    delta = {}

    for forward_iter in range(maxit_f):
        print("forward_iter: ", forward_iter)

        k = k+1

        # forward step
        zero_coeffs = np.where(w[k-1] == 0)[0]
        if len(zero_coeffs)==0: return w[k-1]
        
        err_after_addition = []
        residual = y - X.dot(w[k-1])
        for i in zero_coeffs:

            if relearn_f:
                F_trial = F[k-1].union({i})
                w_added = np.zeros((d,1))
                w_added[list(F_trial)] = np.linalg.lstsq(X[:, list(F_trial)], y, rcond=None)[0]
                
            else:
                # Per figure 3 line 8 in paper, do not retrain old variables.
                # Only look for optimal alpha, which is solving for new w iff X is unitary
                alpha = X[:,i].T.dot(residual)/np.linalg.norm(X[:,i])**2
                w_added = np.copy(w[k-1])
                w_added[i] = alpha

            err_after_addition.append(np.linalg.norm(X.dot(w_added)-y))
        i = zero_coeffs[np.argmin(err_after_addition)]
        
        F[k] = F[k-1].union({i})
        w[k] = np.zeros((d,1), dtype=np.complex64)
        w[k][list(F[k])] = np.linalg.lstsq(X[:, list(F[k])], y,rcond=None)[0]

        # check for break condition
        delta[k] = np.linalg.norm(X.dot(w[k-1]) - y) - np.linalg.norm(X.dot(w[k]) - y)
        if delta[k] < epsilon: return w[k-1]

        # backward step, do once every few forward steps
        if forward_iter % backwards_freq == 0 and forward_iter > 0:

            for backward_iter in range(maxit_b):

                non_zeros = np.where(w[k] != 0)[0]
                err_after_simplification = []
                for j in non_zeros:

                    if relearn_b:
                        F_trial = F[k].difference({j})
                        w_simple = np.zeros((d,1))
                        w_simple[list(F_trial)] = np.linalg.lstsq(X[:, list(F_trial)], y, rcond=None)[0]
                
                    else:
                        w_simple = np.copy(w[k])
                        w_simple[j] = 0

                    err_after_simplification.append(np.linalg.norm(X.dot(w_simple) - y))
                j = np.argmin(err_after_simplification)
                w_simple = np.copy(w[k])
                w_simple[non_zeros[j]] = 0

                # check for break condition on backward step
                delta_p = err_after_simplification[j] - np.linalg.norm(X.dot(w[k]) - y)
                if delta_p > 0.5*delta[k]: break

                k = k-1;
                F[k] = F[k+1].difference({j})
                w[k] = np.zeros((d,1))
                w[k][list(F[k])] = np.linalg.lstsq(X[:, list(F[k])], y,rcond=None)[0]

    return w[k] 
    
