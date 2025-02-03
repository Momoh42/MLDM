import os
import numpy as np
import cv2
import time
import numpy as np
import datetime
from scipy.stats import entropy
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import random

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

start,end,step=0,49,1
deg=3

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
    result=result[:,:image_height,:]
    return result

def crop_image(image, x_size, y_size, x_upper_left, y_upper_left):
    return image[x_upper_left:x_upper_left+x_size,y_upper_left:y_upper_left+y_size ]

def get_centered_initial_coordinate(images, x_size, y_size):
    x_coord_t_0 = images[0].shape[0]/2-x_size/2
    y_coord_t_0 =images[0].shape[1]/2-y_size/2
    return int(x_coord_t_0), int(y_coord_t_0) 


#array_image_folder_0_20_2=loading_array_npy(npy_folder_0_20_2)
array_image_folder_0_22_30=loading_array_npy(npy_folder_0_22_30)
#array_image_folder_0_24_30=loading_array_npy(npy_folder_0_24_30)

x_upper_left, y_upper_left=get_centered_initial_coordinate(array_image_folder_0_22_30,x_size,y_size)

id_image=np.arange(start,end+1,step=step)

#centered_array_image_folder_0_20_2=array_image_folder_0_20_2[id_image,:,:]
centered_array_image_folder_0_22_30=array_image_folder_0_22_30[id_image,:,:]
#centered_array_image_folder_0_24_30=array_image_folder_0_24_30[id_image,:,:]

#centered_array_image_folder_0_20_2=centered_array_image_folder_0_20_2/255
centered_array_image_folder_0_22_30=centered_array_image_folder_0_22_30/255
#centered_array_image_folder_0_24_30=centered_array_image_folder_0_24_30/255

#print(array_image_folder_0_20_2.shape)
print(array_image_folder_0_22_30.shape)
#print(array_image_folder_0_24_30.shape)

#print(centered_array_image_folder_0_20_2.shape)
print(centered_array_image_folder_0_22_30.shape)
#print(centered_array_image_folder_0_24_30.shape)

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

def parallel_lowess(U,random_range,lowess_result):
    new_U,matrix_move=get_square_U_aling(U,random_range,x_size,y_size,x_upper_left,y_upper_left)
    entropy_img=calc_images_n_entropy(new_U)

    fitness=mean_squared_error(entropy_img,lowess_result[:, 1])
    return fitness,matrix_move

def shaked_cropped_square(U,random_range,nb_of_shake):
    now = datetime.datetime.now()
    directory=os.getcwd()+"/"+str(now.day)+"_"+str(now.month)+"_"+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_"+str(random_range)
    os.makedirs(directory, exist_ok=True)
    array_fitness=np.empty((nb_of_shake))
    array_matrix_move=np.empty((nb_of_shake,int(end-start),2))
    #array_coef=np.empty((nb_of_shake,deg+1))


    entropy_img_patches=calc_entropy_labdata_patches(U,is_it_transposed=False)
    entropy_img_mean=np.mean(entropy_img_patches,axis=(1,2))

    x_val_lowess, lowess_result=calc_and_plot_lowess_for_lab_data_ent(entropy_img_mean,start,end,0.2)

    for i in range(nb_of_shake):

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
    return directory

begin = time.time()
directory=shaked_cropped_square(centered_array_image_folder_0_22_30,500,5000)
ending = time.time()
print(ending - begin)

random_range=50

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

def get_square_U(U_matrix,array_move,x_size,y_size,x_upper_left,y_upper_left):
    rows=array_move[:,0]
    cols=array_move[:,1]

    new_U = np.empty((U_matrix.shape[0],x_size,y_size))
    new_U[0,:]=U_matrix[0,x_upper_left:x_upper_left+x_size,y_upper_left:+y_upper_left+y_size]
    for i in range(1,U_matrix.shape[0]):
        new_U[i,:] = U_matrix[i,x_upper_left+rows[i-1]:x_upper_left+rows[i-1]+x_size,x_upper_left+cols[i-1]:x_upper_left+cols[i-1]+x_size]

    new_U=new_U.reshape(new_U.shape[0],-1)
    return new_U

def genetic_lowess(U,new_move,lowess_result):
    new_U=get_square_U(U,new_move,x_size,y_size,x_upper_left,y_upper_left)
    entropy_img=calc_images_n_entropy(new_U)

    fitness=mean_squared_error(entropy_img,lowess_result[:, 1])
    return fitness,new_move

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
    #old_fitness=array_fitness

    entropy_img_patches=calc_entropy_labdata_patches(U,is_it_transposed=False)
    entropy_img_mean=np.mean(entropy_img_patches,axis=(1,2))

    x_val_lowess, lowess_result=calc_and_plot_lowess_for_lab_data_ent(entropy_img_mean,start,end,frac=0.2)

    for i in range(nb_generation):
        if(i<(nb_generation*1/5)):
            new_move=new_generation(array_move,arg_sort,True)
        else:
            new_move=new_generation(array_move,arg_sort,False)

        #new_coef=[]
        new_array_fitness=[]
        new_array_move=[]

        for move in new_move:
            fitness,matrix_move=genetic_lowess(U,move,lowess_result)
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
        #old_fitness=new_array_fitness

genetic_algo(directory,800,centered_array_image_folder_0_22_30)  