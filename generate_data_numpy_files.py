import glob
from tqdm import tqdm
import numpy as np

print("Loading Files from Test and Train Folders")
testing_files, training_files = glob.glob("test/*"), glob.glob("train/*")

print("Creating Test Images Numpy Array")
test_images = []
for image in tqdm(testing_files):
    im = np.load(image)
    test_images.append(im)
test_images = np.array(test_images)
test_images = np.expand_dims(test_images, 0)
test_images = np.concatenate(test_images)
np.save('test_images',test_images)


print("Creating Train Images Numpy Array (this will take a while because it's a big array)")
train_images = []
for image in tqdm(training_files):
    im = np.load(image)
    train_images.append(im)
train_images = np.array(train_images)
train_images = np.expand_dims(train_images, 0)
train_images = np.concatenate(train_images)

print("Trained Images Should Be: and are: (645, 262144, 31) " + str(train_images.shape))
np.save('train_images', train_images)
