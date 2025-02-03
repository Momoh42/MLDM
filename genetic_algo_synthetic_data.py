import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import scipy.stats as stats
import random
import warnings
from sklearn.linear_model import LinearRegression
import copy
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import entropy
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import re

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

random_range=2


def get_centered_initial_coordinate(images, x_size, y_size):
    x_coord_t_0 = images[0].shape[0]/2-x_size/2
    y_coord_t_0 =images[0].shape[1]/2-y_size/2
    return int(x_coord_t_0), int(y_coord_t_0) 

def loading_array_npy(file_name):
    result=np.load(file_name)
    result=result[:,:image_height,:]
    return result

A=np.array(np.load(file_A,allow_pickle=True)["solution_rescaled"])

x_upper_left, y_upper_left=get_centered_initial_coordinate(A,x_size,y_size)

id_image=np.arange(start,end+1,step)

centered_A=A[id_image,:,:]

print(centered_A.shape)

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
        for j in range(len(array_move[i])):
            if random.random()<mutate_value:
                array_result[i,j]=array_move[i,j]+np.random.randint(-random_range, random_range+1, size=(1,2))
                if(array_result[i,j][0]+x_upper_left>image_height-x_size):
                    array_result[i,j][0]=image_height-x_size-x_upper_left

                if(array_result[i,j][0]+x_upper_left<0):
                    array_result[i,j][0]=-x_upper_left

                if(array_result[i,j][1]+y_upper_left>image_height-y_size):
                    array_result[i,j][1]=image_height-y_size-y_upper_left
                
                if(array_result[i,j][1]+y_upper_left<0):
                    array_result[i,j][0]=-y_upper_left
            else:
                array_result[i,j]=array_move[i,j]

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

def fitness_entropy(arr_entropy,start,deg,weight=1): #better if the entropy is a np array shape (-1,1)
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
        list_of_entropies[i].append(entropy(list_of_images3Darrays[i]))
    return np.array(list_of_entropies)

def parrallel_entropy(U,new_move):
    new_U=get_square_U(U,new_move,x_size,y_size,x_upper_left,y_upper_left)
    entropy_img=calc_images_n_entropy(new_U)
    fitness, nro_min, coefs_with_int_dec, y_values, x_values=fitness_entropy(entropy_img,start,deg=deg,weight=1)

    return fitness,new_move,coefs_with_int_dec

def parallel_lowess(U,new_move,lowess_result):
    new_U=get_square_U(U,new_move,x_size,y_size,x_upper_left,y_upper_left)
    entropy_img=calc_images_n_entropy(new_U)

    fitness=mean_squared_error(entropy_img,lowess_result[:, 1])
    return fitness,new_move

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

def genetic_algo(folder,nb_generation,U):
    now = datetime.datetime.now()
    directory=folder+"/Generation_"+str(now.day)+"_"+str(now.month)+"_"+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)
    os.makedirs(directory, exist_ok=True)

    list_directory=[x for x in os.listdir(folder) if x.endswith(".npy") and x.startswith("Move")]
    list_directory=[os.path.join(folder, element) for element in list_directory]
    array_move=np.load(list_directory[0],allow_pickle=True)

    list_directory=[x for x in os.listdir(folder) if x.endswith(".npy") and x.startswith("Fitness")]
    list_directory=[os.path.join(folder, element) for element in list_directory]
    array_fitness=np.load(list_directory[0],allow_pickle=True)

    #arg_sort=np.flip(np.argsort(array_fitness))
    arg_sort=np.argsort(array_fitness)
    old_fitness=array_fitness

    square_align=np.load("Miss_aligned.npy")
    miss_aligned_data=get_square_U(U, square_align,x_size,y_size,x_upper_left,y_upper_left)

    entropy_img=calc_images_n_entropy(miss_aligned_data)
    x_val_lowess, lowess_result=calc_and_plot_lowess_for_lab_data_ent(entropy_img[:,0],start,end, frac=0.1)

    for i in range(nb_generation):
        if(i<(nb_generation*1/5)):
            new_move=new_generation(array_move,arg_sort,True)
        else:
            new_move=new_generation(array_move,arg_sort,False)

        #new_coef=[]
        new_array_fitness=[]
        new_array_move=[]

        for move in new_move:
            fitness,matrix_move=parallel_lowess(U,move,lowess_result)
            #fitness,matrix_move,coefs_with_int_dec=parrallel_entropy(U,move)
            new_array_fitness.append(fitness)
            new_array_move.append(matrix_move)
            #new_coef.append(coefs_with_int_dec)

        new_array_fitness=np.array(new_array_fitness)
        new_array_move=np.array(new_array_move)
        #new_coef=np.array(new_coef)

        #save_array(directory+"/GEN_Coef_"+str(i+1)+".npy",new_coef)
        save_array(directory+"/GEN_Fitness_"+str(i+1)+".npy",new_array_fitness)
        save_array(directory+"/GEN_Move_"+str(i+1)+".npy",new_array_move)


        """if(np.abs(np.mean(old_fitness) -np.mean(old_fitness))<0.00001):
            break"""

        #arg_sort=np.flip(np.argsort(new_array_fitness))
        arg_sort=np.argsort(new_array_fitness)
        array_move=new_array_move
        old_fitness=new_array_fitness

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def loading_file_npy_folder(path,begin):
    list_directory=[x for x in os.listdir(path) if x.endswith(".npy") and x.startswith(begin)]
    list_directory.sort(key=natural_keys)
    list_directory=[os.path.join(path, element) for element in list_directory]

    return list_directory

def restart_gen(directory,nb_generation,U):
    list_directory_fitness=loading_file_npy_folder(directory,"GEN_Fitness")[-1]
    list_directory_Move=loading_file_npy_folder(directory,"GEN_Move")[-1]

    filename = os.path.basename(list_directory_Move)
    numbers = int(re.findall(r'\d+', filename)[0])

    array_fitness=np.load(list_directory_fitness,allow_pickle=True)
    array_fitness=np.array(array_fitness)
    
    array_move=np.load(list_directory_Move,allow_pickle=True)
    array_move=np.array(array_move)

    arg_sort=np.argsort(array_fitness)

    square_align=np.load("Miss_aligned.npy")
    miss_aligned_data=get_square_U(U, square_align,x_size,y_size,x_upper_left,y_upper_left)

    entropy_img=calc_images_n_entropy(miss_aligned_data)
    x_val_lowess, lowess_result=calc_and_plot_lowess_for_lab_data_ent(entropy_img[:,0],start,end, frac=0.5)

    for i in range(numbers,nb_generation):
        if(i<(nb_generation*1/5)):
            new_move=new_generation(array_move,arg_sort,True)
        else:
            new_move=new_generation(array_move,arg_sort,False)

        #new_coef=[]
        new_array_fitness=[]
        new_array_move=[]

        for move in new_move:
            fitness,matrix_move=parallel_lowess(U,move,lowess_result)
            #fitness,matrix_move,coefs_with_int_dec=parrallel_entropy(U,move)
            new_array_fitness.append(fitness)
            new_array_move.append(matrix_move)
            #new_coef.append(coefs_with_int_dec)

        new_array_fitness=np.array(new_array_fitness)
        new_array_move=np.array(new_array_move)
        #new_coef=np.array(new_coef)

        #save_array(directory+"/GEN_Coef_"+str(i+1)+".npy",new_coef)
        save_array(directory+"/GEN_Fitness_"+str(i+1)+".npy",new_array_fitness)
        save_array(directory+"/GEN_Move_"+str(i+1)+".npy",new_array_move)


        """if(np.abs(np.mean(old_fitness) -np.mean(old_fitness))<0.00001):
            break"""

        #arg_sort=np.flip(np.argsort(new_array_fitness))
        arg_sort=np.argsort(new_array_fitness)
        array_move=new_array_move
        old_fitness=new_array_fitness

print("A dataset")
genetic_algo(os.getcwd()+"/A_18_12_2023_22_2_5_40",800,centered_A)  