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
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import copy
from scipy.stats import entropy
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

file_A="A_og_sol_and_rescaled_r0_5s2SH_sol2023-12-11_16_18_42.pkl"
file_B="B_og_sol_and_rescaled_r0_1s2_2SH_sol2023-12-11_18_43_26.pkl"
file_C="C_og_sol_and_rescaled_r0s2SH_sol2023-12-11_18_19_18.pkl"
file_D="D_og_sol_and_rescaled_r0_5s1SH_sol2023-12-11_20_43_49.pkl"

image_height=320
image_width=320
nb_img=60

x_size, y_size=200,200

start,end,step=0,479,8
deg=3

def loading_array_npy(file_name):
    result=np.load(file_name)
    result=result[:,:image_height,:]
    return result

def crop_image(image, x_size, y_size, x_upper_left, y_upper_left):
    return image[x_upper_left:x_upper_left+x_size,y_upper_left:y_upper_left+y_size ]

def get_centered_initial_coordinate(images, x_size, y_size):
    x_coord_t_0 = images[0].shape[0]/2-x_size/2
    y_coord_t_0 =images[0].shape[1]/2-y_size/2
    return int(x_coord_t_0), int(y_coord_t_0) 

A=np.array(np.load(file_A,allow_pickle=True)["solution_rescaled"])

x_upper_left, y_upper_left=get_centered_initial_coordinate(A,x_size,y_size)

id_image=np.arange(start,end+1,step)

centered_A=A[id_image,:,:]

print(centered_A.shape)

def save_array(file_name,array):
    with open(file_name,'wb') as f:
            np.save(f,array)
    f.close()


def get_square_U_aling(U_matrix, random_range,x_size,y_size,x_upper_left,y_upper_left):
    rows = np.random.randint(-random_range,random_range+1, size=U_matrix.shape[0]-1)
    cols = np.random.randint(-random_range, random_range+1, size=U_matrix.shape[0]-1)
    square_align=np.array(list(zip(rows, cols)))

    new_U = np.empty((U_matrix.shape[0],x_size,y_size))
    new_U[0,:]=U_matrix[0,x_upper_left:x_upper_left+x_size,y_upper_left:y_upper_left+y_size]

    for i in range(1,U_matrix.shape[0]):
        new_U[i,:] = U_matrix[i,x_upper_left+rows[i-1]:x_upper_left+rows[i-1]+x_size,x_upper_left+cols[i-1]:x_upper_left+cols[i-1]+x_size]

    new_U=new_U.reshape(new_U.shape[0],-1)
    return new_U,square_align

def get_square_U(U_matrix,array_move,x_size,y_size,x_upper_left,y_upper_left):
    rows=array_move[:,0]
    cols=array_move[:,1]

    new_U = np.empty((U_matrix.shape[0],x_size,y_size))
    new_U[0,:]=U_matrix[0,x_upper_left:x_upper_left+x_size,y_upper_left:+y_upper_left+y_size]
    for i in range(1,U_matrix.shape[0]):
        new_U[i,:] = U_matrix[i,x_upper_left+rows[i-1]:x_upper_left+rows[i-1]+x_size,x_upper_left+cols[i-1]:x_upper_left+cols[i-1]+x_size]

    new_U=new_U.reshape(new_U.shape[0],-1)
    return new_U

def calc_images_n_entropy(list_of_images3Darrays):
    list_of_entropies =  [[] for _ in range(len(list_of_images3Darrays))]
    for i in range(len(list_of_images3Darrays)):
        list_of_entropies[i].append(entropy(list_of_images3Darrays[i]))
    return np.array(list_of_entropies)

def calc_entropy_labdata_patches(lab_data, size=200, is_it_transposed=True):
    if is_it_transposed:
        lab_data = np.transpose(lab_data,(2,0,1))
    dim1_lims = np.arange(0,lab_data.shape[1]-size,size)
    dim2_lims = np.arange(0,lab_data.shape[2]-size,size)
    #print(dim1_lims)
    #print(len(dim1_lims))
    #print(dim2_lims)
    #print(len(dim2_lims))
    
    counter_a = 0
    counter_b = 0
    counter_c = 0
    entrops_patches = np.empty((lab_data.shape[0],len(dim1_lims),len(dim2_lims)))
    #print(entrops_patches.shape)
    for img in lab_data:
        counter_b=0
        for i in dim1_lims:
            counter_c=0
            for j in dim2_lims:
                #print(i,j)
                entrops_patches[counter_a,counter_b,counter_c] = entropy(img[i:i+size,j:j+size].flatten())
                #print(counter_c)
                counter_c +=1
            counter_b +=1
        counter_a +=1
    return entrops_patches

def calc_and_plot_lowess_for_lab_data_ent(ent_1darray, start, end, frac=0.3):
    x_val_lowess = np.arange(start,(end+1)/step)
    x_val_ent = np.arange(len(ent_1darray))
    lowess_result = sm.nonparametric.lowess(ent_1darray, x_val_lowess, frac=frac)
    return x_val_lowess, lowess_result

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
    fitness, nro_min, coefs_with_int_dec, y_values, x_values=fitness_entropy(entropy_img,start,deg=deg,weight=1)

        #if(nro_min==1):
            #running=False
    return fitness,matrix_move,coefs_with_int_dec

def parallel_lowess(U,random_range,lowess_result):
    new_U,matrix_move=get_square_U_aling(U,random_range,x_size,y_size,x_upper_left,y_upper_left)
    entropy_img=calc_images_n_entropy(new_U)

    fitness=mean_squared_error(entropy_img,lowess_result[:, 1])
    return fitness,matrix_move

def shaked_cropped_square(U,random_range,nb_of_shake):
    now = datetime.datetime.now()
    directory=os.getcwd()+"/A_"+str(now.day)+"_"+str(now.month)+"_"+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_"+str(random_range)
    os.makedirs(directory, exist_ok=True)
    array_fitness=np.empty((nb_of_shake))
    array_matrix_move=np.empty((nb_of_shake,int(end/step-start),2))
    array_coef=np.empty((nb_of_shake,deg+1))

    """entropy_real=calc_images_n_entropy(U[:,x_upper_left:x_upper_left+x_size,y_upper_left:y_upper_left+y_size].reshape(U.shape[0],x_size*y_size))
    x_val_lowess, lowess_real=calc_and_plot_lowess_for_lab_data_ent(entropy_real[:,0],start,end, frac=0.5)
"""
    square_align=np.load(os.getcwd()+"/Miss_aligned.npy")
    miss_aligned_data=get_square_U(U, square_align,x_size,y_size,x_upper_left,y_upper_left)

    #miss_aligned_data,matrix_move=get_square_U_aling(U,random_range,x_size,y_size,x_upper_left,y_upper_left)

    entropy_img=calc_images_n_entropy(miss_aligned_data)
    x_val_lowess, lowess_result=calc_and_plot_lowess_for_lab_data_ent(entropy_img[:,0],start,end, frac=0.5)
    """plt.plot(lowess_result[:,1],label="miss lowess")
    plt.plot(lowess_real[:,1],label="r lowess")
    plt.scatter(range(len(entropy_img[:,0])),entropy_img[:,0],label="miss entropy",alpha=0.3)
    plt.scatter(range(len(entropy_real[:,0])),entropy_real[:,0],label="r entropy",alpha=0.3)
    plt.legend()
    plt.show()"""

    for i in range(nb_of_shake):
        print(i)
        fitness,matrix_move=parallel_lowess(U,random_range,lowess_result)
        array_fitness[i]=fitness
        array_matrix_move[i]=matrix_move
        #array_coef[i]=coefs_with_int_dec
        #,coefs_with_int_dec

        """if (i+1)%nb_interval==0:
            list_directory=[x for x in os.listdir(directory) if x.endswith(".npy") and x.startswith("sparsity")]
            list_directory=[os.path.join(directory, element) for element in list_directory]
            tmp=[]
            for file in list_directory:
                tmp.append(np.load(file,allow_pickle=True))
            tmp=np.array(tmp)

            save_array(directory+"/Sparsity_"+str(i-nb_interval+2)+"_"+str(i+1)+".npy",tmp)

            for file in list_directory:
                os.remove(file)

            list_directory=[x for x in os.listdir(directory) if x.endswith(".npy") and x.startswith("move")]
            list_directory=[os.path.join(directory, element) for element in list_directory]
            tmp=[]
            for file in list_directory:
                tmp.append(np.load(file,allow_pickle=True))
            tmp=np.array(tmp)

            save_array(directory+"/Move_"+str(i-nb_interval+2)+"_"+str(i+1)+".npy",tmp)

            for file in list_directory:
                os.remove(file)"""
                
    #save_array(directory+"/Coef_1_"+str(nb_of_shake)+".npy",array_coef)
    save_array(directory+"/Fitness_1_"+str(nb_of_shake)+".npy",array_fitness)
    save_array(directory+"/Move_1_"+str(nb_of_shake)+".npy",array_matrix_move)

begin = time.time()
print("A dataset")
shaked_cropped_square(centered_A,40,5000)
ending = time.time()

print(ending - begin)