from joblib import Parallel, delayed,parallel_backend
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
from scipy.stats import entropy
import pysindy as ps
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import datetime
import scipy.stats as stats
import random
import warnings
import copy

warnings.filterwarnings('error',category=RuntimeWarning)
warnings.filterwarnings('error',category=UserWarning)

folder_0_20_2="./explore-2023-07/file/2023-07_n1-to-n50-as-jpg/0.20__2"
folder_0_22_30="./explore-2023-07/file/2023-07_n1-to-n50-as-jpg/0.22__30"
folder_0_24_30="./explore-2023-07/file/2023-07_n1-to-n50-as-jpg/0.24__30"

npy_folder_0_20_2="0.20__2.npy"
npy_folder_0_22_30="0.22__30.npy"
npy_folder_0_24_30="0.24__30.npy"

image_height=1920
image_width=2560
nb_img=50

x_size, y_size=200,200

start,end=5,45

def loading_image_folder(path):
    result=[x for x in os.listdir(path) if x.endswith(".jpg") ]
    result=sorted(result, key=lambda x: int(x.partition('.')[0]))
    result = [os.path.join(path, element) for element in result]
    return result

def loading_array_folder(path):
    image_folder=loading_image_folder(path)
    array_image_folder=np.empty((50,image_height,image_width), dtype=object)
    for i in range(len(image_folder)):
        tmp = cv2.imread(image_folder[i],flags=cv2.IMREAD_GRAYSCALE)
        array_image_folder[i]=tmp[:image_height]
        
    return array_image_folder

def loading_array_npy(file_name):
    result=np.load(file_name)
    result=result[:,:1920,:]
    return result

def crop_image(image, x_size, y_size, x_upper_left, y_upper_left):
    return image[x_upper_left:x_upper_left+x_size,y_upper_left:y_upper_left+y_size ]

def get_centered_initial_coordinate(images, x_size, y_size):
    x_coord_t_0 = images[0].shape[0]/2-x_size/2
    y_coord_t_0 =images[0].shape[1]/2-y_size/2
    return int(x_coord_t_0), int(y_coord_t_0) 

array_image_folder_0_20_2=loading_array_npy(npy_folder_0_20_2)
array_image_folder_0_20_30=loading_array_npy(npy_folder_0_22_30)
array_image_folder_0_24_30=loading_array_npy(npy_folder_0_24_30)

x_upper_left, y_upper_left=get_centered_initial_coordinate(array_image_folder_0_20_2,x_size,y_size)

id_image=np.arange(start,end+1)

centered_array_image_folder_0_20_2=array_image_folder_0_20_2[id_image,:,:]
centered_array_image_folder_0_20_30=array_image_folder_0_20_30[id_image,:,:]
centered_array_image_folder_0_24_30=array_image_folder_0_24_30[id_image,:,:]

centered_array_image_folder_0_20_2=centered_array_image_folder_0_20_2/255
centered_array_image_folder_0_20_30=centered_array_image_folder_0_20_30/255
centered_array_image_folder_0_24_30=centered_array_image_folder_0_24_30/255

def save_array(file_name,array):
    with open(file_name,'wb') as f:
            np.save(f,array)
    f.close()


def get_square_U_aling(U_matrix, random_range,x_size,y_size,x_upper_left,y_upper_left):
    rows = np.random.randint(-random_range, random_range+1, size=U_matrix.shape[0]-1)
    cols = np.random.randint(-random_range, random_range+1, size=U_matrix.shape[0]-1)
    square_align=np.array(list(zip(rows, cols)))

    new_U = np.empty((U_matrix.shape[0],x_size,y_size))
    new_U[0,:]=U_matrix[0,x_upper_left:x_upper_left+x_size,y_upper_left:y_upper_left+y_size]

    for i in range(1,U_matrix.shape[0]):
        new_U[i,:] = U_matrix[i,x_upper_left+rows[i-1]:x_upper_left+rows[i-1]+x_size,x_upper_left+cols[i-1]:x_upper_left+cols[i-1]+x_size]

    new_U=new_U.reshape(new_U.shape[0],-1)
    return new_U,square_align

def parrallel_PCA_pysindy(U,random_range,n_components):
    scaler = StandardScaler()
    pca=PCA(n_components=n_components,svd_solver="randomized",random_state=42)

    poly_order = 2
    threshold = 0.004
    model = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold),
                feature_library=ps.PolynomialLibrary(degree=poly_order))
    running=True

    nb_error=0
    while running:
        new_U,matrix_move=get_square_U_aling(U,random_range,x_size,y_size,x_upper_left,y_upper_left)
        new_U_transfrom=scaler.fit_transform(new_U)
        pca.fit(new_U_transfrom)
        eigen_vectors=pca.components_.T

        try:
            model.fit(eigen_vectors, t=1)
            coef=model.coefficients()


            x0_train=eigen_vectors[0]
            t_train = np.arange(0, 100, 1)

            x_sim = model.simulate(x0_train, t_train)
            sparsity=np.linalg.norm(coef)+np.linalg.norm(coef, ord=1)
            if(np.count_nonzero(coef)>0):
                break
        except UserWarning as w:
            running=True
            nb_error+=1
        except ValueError as v:
            running=True
            nb_error+=1
        except DeprecationWarning as w:
            running=True
            nb_error+=1
        except RuntimeWarning as w:
            running=True
            nb_error+=1
        except RuntimeError as w:
            running=True
            nb_error+=1
        else:
            running=False
    #print(nb_error)
    return coef,sparsity,matrix_move
    
def calc_images_n_entropy(list_of_images3Darrays):
    list_of_entropies =  [[] for _ in range(len(list_of_images3Darrays))]
    for i in range(len(list_of_images3Darrays)):
        list_of_entropies[i].append(entropy(list_of_images3Darrays[i]))
    return np.array(list_of_entropies)

def fitness_entropy(arr_entropy,start,deg,weight): #better if the entropy is a np array shape (-1,1)
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

def parrallel_entropy(U,random_range):
    #running=True
    new_U,matrix_move=get_square_U_aling(U,random_range,x_size,y_size,x_upper_left,y_upper_left)
    entropy_img=calc_images_n_entropy(new_U)
    fitness, nro_min, coefs_with_int_dec, y_values, x_values=fitness_entropy(entropy_img,start,deg=3,weight=1)

        #if(nro_min==1):
            #running=False
    return fitness,matrix_move

def shaked_cropped_square(U,random_range,nb_of_shake):
    now = datetime.datetime.now()
    directory=os.getcwd()+"/"+str(now.day)+"_"+str(now.month)+"_"+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_"+str(random_range)
    os.makedirs(directory, exist_ok=True)

    matrix_move=np.empty((nb_of_shake,U.shape[0]-1,2))
    
    results = Parallel(n_jobs=-1)(
        delayed(parrallel_entropy)(U,random_range) for i in range(nb_of_shake)
    )

    #models=[]
    fitness=[]
    matrix_move=[]
    for i in range(len(results)):
        #models.append(results[i][0])
        fitness.append(results[i][0])
        matrix_move.append(results[i][1])

    fitness=np.array(fitness)
    #models=np.array(models)
    matrix_move=np.array(matrix_move)
    #save_array(directory+"/Model_1_"+str(nb_of_shake)+".npy",models)
    save_array(directory+"/Fitness_1_"+str(nb_of_shake)+".npy",fitness)
    save_array(directory+"/Move_1_"+str(nb_of_shake)+".npy",matrix_move)

start = time.time()
shaked_cropped_square(centered_array_image_folder_0_20_2,300,10)
end = time.time()
print(end - start)