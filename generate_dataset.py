import os
import glob
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np
test, train= glob.glob("raw_images/*")[:20], glob.glob("raw_images/*")[20:]

print(test)
def flatten(matrix):
    array = np.split(matrix, 512, axis=1)
    array = [np.squeeze(i) for i in array]
    return np.array(np.concatenate(array, 0))

#create folders
os.makedirs("test")
os.makedirs("train")

# Generate test data from 20 images stored in four numpy files
i = 0
for image in tqdm(test):
    full_matrix = loadmat(image, verify_compressed_data_integrity=False)['ref']
    np.save('test/img_0_0_' + str(i), flatten(full_matrix[:512,:512]))
    np.save('test/img_0_1_' + str(i), flatten(full_matrix[512:1024,:512]))
    np.save('test/img_1_0_' + str(i), flatten(full_matrix[:512,512:1024]))
    np.save('test/img_1_1_' + str(i), flatten(full_matrix[512:1024,512:1024]))

i = 0
# generate training data from the rest 
for image in tqdm(train):
    full_matrix = loadmat(image, verify_compressed_data_integrity=False)['ref']
    np.save('train/img_0_0_' + str(i), flatten(full_matrix[:512,:512]))
    np.save('train/img_0_0_flip_x_' + str(i), flatten(np.flip(full_matrix[:512,:512],0)))
    np.save('train/img_0_0_flip_y_' + str(i), flatten(np.flip(full_matrix[:512,:512],1)))
    np.save('train/img_0_1_' + str(i), flatten(full_matrix[512:1024,:512]))
    np.save('train/img_0_1_flip_x_' + str(i), flatten(np.flip(full_matrix[512:1024,:512],0)))
    np.save('train/img_0_1_flip_y_' + str(i), flatten(np.flip(full_matrix[512:1024,:512],1)))
    np.save('train/img_1_0_' + str(i), flatten(full_matrix[:512,512:1024]))
    np.save('train/img_1_0_flip_x_' + str(i), flatten(np.flip(full_matrix[:512,512:1024],0)))
    np.save('train/img_1_0_flip_y_' + str(i), flatten(np.flip(full_matrix[:512,512:1024],1)))
    np.save('train/img_1_1_' + str(i), flatten(full_matrix[512:1024,512:1024]))
    np.save('train/img_1_1_flip_x_' + str(i), flatten(np.flip(full_matrix[512:1024,512:1024],0)))
    np.save('train/img_1_1_flip_y_' + str(i), flatten(np.flip(full_matrix[512:1024,512:1024],1)))
    i+=1
