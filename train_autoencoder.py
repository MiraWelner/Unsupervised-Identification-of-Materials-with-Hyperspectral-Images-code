import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.losses import mean_squared_error
import numpy as np


EPOCHS = 50
BATCH_SIZE = 1
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00015)

def normalized_mse(y_true, y_pred):
    """
    Create a custom loss function since keras doesn't have normalized mse
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    return tf.divide(mean_squared_error(y_true, y_pred),
                      mean_squared_error(y_true, y_true*0))

# The autoencoder
input_img = Input(shape=(262144, 31))

#Use the map dimentions as channels, compress the spectra until you have the input number of spectra
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(input_img)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(2), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = MaxPooling1D(pool_size=(1), strides=(3), padding='valid')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Conv1D(31, (3), activation='relu', padding='same')(encoded_spectra)
encoded_spectra = Dense(31, activation='relu')(encoded_spectra)
encoded_spectra = Dense(31, activation='relu')(encoded_spectra)
encoded_spectra = Dropout(0.2)(encoded_spectra)
encoded_spectra = Dense(31, activation='relu')(encoded_spectra)
encoded_spectra = Dense(31, activation='relu')(encoded_spectra)


## The decoder uses matrix manipulations rather than neural networks
pseudo_inverse = tf.linalg.pinv(encoded_spectra)
material_maps = tf.linalg.matmul(input_img, pseudo_inverse)
positive_forced = tf.nn.relu(material_maps, name ='ReLU')
final_hyperspectral =  tf.linalg.matmul(positive_forced, encoded_spectra)

autoencoder= Model(input_img,final_hyperspectral)

train_images = np.load('train_images.npy')
test_images = np.load('test_images.npy')


# Train the autoencoder model
autoencoder.compile(optimizer=optimizer, loss=normalized_mse)
history = autoencoder.fit(train_images, train_images,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(test_images, test_images),
                verbose=1)
autoencoder.save('saved_autoencoder')


# Display Training Results
print("Training Loss: " + str(round(history.history['loss'][EPOCHS-1],3)))
print("Testing Loss: " + str(round(history.history['val_loss'][EPOCHS-1],3)))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss of the Hyperspectral Image Simulator over Time')
plt.ylabel('Mean Squared Logarithmic Error')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("training.png")
