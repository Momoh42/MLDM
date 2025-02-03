import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import linear_damped_SHO
from pysindy.utils import cubic_damped_SHO
from pysindy.utils import linear_3D
from pysindy.utils import hopf
from pysindy.utils import lorenz
import pysindy as ps
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import StandardScaler
import datetime
import scipy.stats as stats
import random
import warnings
from joblib import Parallel, delayed,parallel_backend
from sklearn.linear_model import LinearRegression
import copy
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import entropy

folder_0_20_2="./explore-2023-07/file/2023-07_n1-to-n50-as-jpg/0.20__2"
folder_0_22_30="./explore-2023-07/file/2023-07_n1-to-n50-as-jpg/0.22__30"
folder_0_24_30="./explore-2023-07/file/2023-07_n1-to-n50-as-jpg/0.24__30"

npy_folder_0_20_2="0.20__2.npy"
npy_folder_0_22_30="0.22__30.npy"
npy_folder_0_24_30="0.24__30.npy"

image_height=1920
image_width=2560
nb_img=50

x_size, y_size=300,300
start,end=5,45

def get_centered_initial_coordinate(images, x_size, y_size):
    x_coord_t_0 = images[0].shape[0]/2-x_size/2
    y_coord_t_0 =images[0].shape[1]/2-y_size/2
    return int(x_coord_t_0), int(y_coord_t_0) 

def loading_array_npy(file_name):
    result=np.load(file_name)
    result=result[:,:image_height,:]
    return result

array_image_folder_0_20_2=loading_array_npy(npy_folder_0_20_2)

x_upper_left, y_upper_left=get_centered_initial_coordinate(array_image_folder_0_20_2,x_size,y_size)

id_image=np.arange(9,31,2)
centered_array_image_folder_0_20_2=array_image_folder_0_20_2[id_image,:,:]

def get_square_U(U_matrix,array_move,x_size,y_size,x_upper_left,y_upper_left):
    rows=array_move[:,0]
    cols=array_move[:,1]

    new_U = np.empty((U_matrix.shape[0],x_size,y_size))
    new_U[0,:]=U_matrix[0,x_upper_left:x_upper_left+x_size,y_upper_left:+y_upper_left+y_size]
    for i in range(1,U_matrix.shape[0]):
        new_U[i,:] = U_matrix[i,x_upper_left+rows[i-1]:x_upper_left+rows[i-1]+x_size,x_upper_left+cols[i-1]:x_upper_left+cols[i-1]+x_size]

    new_U=new_U.reshape(new_U.shape[0],-1)
    return new_U

def new_generation(array_move,arg_sort,begin):
    mutate_value=0.2
    value_crossover=0.5
    array_result=np.empty((array_move.shape[0],array_move.shape[1],2),dtype=int)
    nb_of_parents=round(array_move.shape[0]*0.4)
    nb_mutate=round(array_move.shape[0]*0.4)
    top=round(array_move.shape[0]*0.4)

    if(begin):
        id_mutate=random.sample(list(arg_sort), nb_mutate)
    else:
        id_mutate=random.sample(list(arg_sort[:top]), nb_mutate)

    id_parents_best=random.sample(list(arg_sort[:top]), round(nb_of_parents*0.8))
    id_parents_worst=random.sample(list(arg_sort[top:]), round(nb_of_parents*0.2))
    id_parents=id_parents_best+id_parents_worst

    array_result[0:nb_mutate,:,:]=create_mutate(np.array([array_move[index] for index in id_mutate ]),mutate_value)
    array_result[nb_mutate:nb_mutate+nb_of_parents,:,:]=create_offsprings_crossover(np.array([array_move[index] for index in id_parents ]),round(value_crossover*array_move.shape[1]))
    array_result[nb_mutate+nb_of_parents:,:]=create_offsprings_avg(np.array([array_move[index] for index in id_parents ]))

    return array_result

def create_mutate(array_move,mutate_value):
    array_result=np.empty_like(array_move)
    for i in range(len(array_move)):
        if random.random()<mutate_value:
            array_result[i]=array_move[i]+np.random.randint(-50, 51, size=(array_move.shape[1],2))
        else:
            array_result[i]=array_move[i]

    return array_result

def create_offsprings_crossover(array_parents,value_crossover):
    array_result=np.empty_like(array_parents)
    for i in [x for x in range(len(array_parents)) if x % 2 == 0]:
        array_result[i,]= np.concatenate((array_parents[i,:value_crossover], array_parents[i+1,value_crossover:]),axis=0)
        array_result[i+1,]= np.concatenate((array_parents[i,value_crossover:], array_parents[i+1,:value_crossover]),axis=0)

    return array_result

def create_offsprings_avg(array_parents):
    array_result=np.empty((int(array_parents.shape[0]/2),array_parents.shape[1],array_parents.shape[2]))
    for i in range(array_result.shape[0]):
        array_result[i,:]= np.mean( np.array([ array_parents[i*2,:],array_parents[i*2+1,:] ]), axis=0 ).astype(int)

    return array_result

def save_array(file_name,array):
    with open(file_name,'wb') as f:
            np.save(f,array)
    f.close()

def fitness_entropy(arr_entropy,start,deg=4,weight=1): #better if the entropy is a np array shape (-1,1)
    """ 
    Calculates the fitness
    The start is the time position of the first image we calculated the entropy from
    """
    x_val = np.array(range(arr_entropy.shape[0])) +start
    #print(x_val)
    poly_features = PolynomialFeatures(degree=deg)
    xpol = poly_features.fit_transform(x_val.reshape(-1,1))
    #print(xpol)
    model4deg = LinearRegression()
    model4deg.fit(xpol, arr_entropy)
    r2 = model4deg.score(xpol, arr_entropy)
    
    coefs_with_int = copy.deepcopy(model4deg.coef_)
    coefs_with_int[0,0] = model4deg.intercept_[0]
    coefs_with_int_dec = np.flip(coefs_with_int).tolist()
    # Calculate how many minimum 
    x_values = np.linspace(start, arr_entropy.shape[0]+start, 20)
    y_values =  np.polyval(coefs_with_int_dec[0], x_values) #coefs in decreasing order
    nro_min =0
    for i in range(1,len(y_values)-1):
        if y_values[i]-y_values[i-1] <0:
            if y_values[i]-y_values[i+1]<0:
                nro_min+=1
    fitness = weight*(nro_min==1)+r2

    return fitness, nro_min, coefs_with_int_dec[0], y_values, x_values

def calc_images_n_entropy(list_of_images3Darrays):
    list_of_entropies =  [[] for _ in range(len(list_of_images3Darrays))]
    for i in range(len(list_of_images3Darrays)):
        for j in list_of_images3Darrays[i]:
            list_of_entropies[i].append(entropy(j))
    return list_of_entropies

def parrallel_entropy(U,new_move):
    new_U=get_square_U(U,new_move,x_size,y_size,x_upper_left,y_upper_left)
    entropy_img=calc_images_n_entropy(new_U)
    fitness, nro_min, coefs_with_int_dec, y_values, x_values=fitness_entropy(entropy_img,start,deg=4,weight=1)

    return fitness,new_move

def parrallel_PCA_pysindy(U,new_move,n_components):
    scaler = StandardScaler()
    pca=PCA(n_components=n_components,svd_solver="randomized",random_state=42)

    poly_order = 2
    threshold = 0.004
    model = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold),
                feature_library=ps.PolynomialLibrary(degree=poly_order))
    
    new_U=get_square_U(U,new_move,x_size,y_size,x_upper_left,y_upper_left)

    new_U_transfrom=scaler.fit_transform(new_U)
    pca.fit(new_U_transfrom)
    eigen_vectors=pca.components_.T

    model.fit(eigen_vectors, t=1)
    coef=model.coefficients()

    x0_train=eigen_vectors[0]
    t_train = np.arange(0, 1000, 1)

    x_sim = model.simulate(x0_train, t_train)
    sparsity=np.linalg.norm(coef)+np.linalg.norm(coef, ord=1)

    return coef,sparsity,new_move

def genetic_algo(folder,nb_generation,U):
    now = datetime.datetime.now()
    directory=folder+"/Generation_"+str(now.day)+"_"+str(now.month)+"_"+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)
    os.makedirs(directory, exist_ok=True)

    list_directory=[x for x in os.listdir(folder) if x.endswith(".npy") and x.startswith("Model")]
    list_directory=[os.path.join(folder, element) for element in list_directory]
    array_coef=np.load(list_directory[0],allow_pickle=True)

    list_directory=[x for x in os.listdir(folder) if x.endswith(".npy") and x.startswith("Move")]
    list_directory=[os.path.join(folder, element) for element in list_directory]
    array_move=np.load(list_directory[0],allow_pickle=True)

    list_directory=[x for x in os.listdir(folder) if x.endswith(".npy") and x.startswith("Sparsity")]
    list_directory=[os.path.join(folder, element) for element in list_directory]
    array_spar=np.load(list_directory[0],allow_pickle=True)

    arg_sort=np.flip(np.argsort(array_spar))
    old_spar=array_spar
    
    for i in range(nb_generation):
        if(i<(nb_generation*1/5)):
            new_move=new_generation(array_move,arg_sort,True)
        else:
            new_move=new_generation(array_move,arg_sort,False)
        new_array_spar=[]

        results = Parallel(n_jobs=-1)(
            delayed(parrallel_entropy)(U,move) for move in new_move
        )

        #models=[]
        new_array_spar=[]
        matrix_move=[]
        for j in range(len(results)):
            #models.append(results[j][0])
            new_array_spar.append(results[j][0])
            matrix_move.append(results[j][1])
        
        new_array_spar=np.array(new_array_spar)
        #models=np.array(models)
        matrix_move=np.array(matrix_move)
        #save_array(directory+"/GEN_Model_"+str(i+1)+".npy",models)
        save_array(directory+"/GEN_Sparsity_"+str(i+1)+".npy",new_array_spar)
        save_array(directory+"/GEN_Move_"+str(i+1)+".npy",matrix_move)

        #print(np.abs(np.mean(new_array_spar) -np.mean(old_spar)))
        if(np.abs(np.mean(new_array_spar) -np.mean(old_spar))<0.00001):
            break

        arg_sort=np.flip(np.argsort(new_array_spar))
        array_move=matrix_move
        old_spar=new_array_spar

genetic_algo(os.getcwd()+"/20_11_2023_19_35_6_5_50",1000,centered_array_image_folder_0_20_2)