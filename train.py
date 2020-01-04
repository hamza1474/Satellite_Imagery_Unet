import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import model_unet
from model_unet import get_unet
import config

seed = config.SEED

#----> Make sure data_numpy.npy is present in data/ <----
print("===========================")
print("FETCHING DATA...")
array = np.load('/data/data_numpy.npy')
print("===========================")


print("===========================")
print("LOADING MODEL...")
model = get_unet()
print("===========================")


print("===========================")
print("PREPARING DATA... (This might take some time)")
img = array[:,:,:,:3]
msk = array[:,:,:,3]
mean = np.mean(img)
std = np.std(img)
img = (img - mean)/std
print("MEAN = " + str(mean), "STD = " + str(std))
print(">>>>>>>>>>>>Update config.py with these values<<<<<<<<<<<")

x_trn, x_val, y_trn, y_val = train_test_split(img, msk,test_size=0.2, random_state=seed)
y_trn = y_trn[:,:,:,None]
y_val = y_val[:,:,:,None]
print("===========================")


print("===========================")
print("PREPARING DATA AUGMENTATION...")
data_gen_args = dict(rotation_range=45.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')  #use 'constant'??


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
X_datagen = ImageDataGenerator(**data_gen_args)
Y_datagen = ImageDataGenerator(**data_gen_args)
X_datagen.fit(x_trn, augment=True, seed=32)
Y_datagen.fit(y_trn, augment=True, seed=32)
X = X_datagen.flow(x_trn, batch_size=16, shuffle=True, seed=32)
Y = Y_datagen.flow(y_trn, batch_size=16, shuffle=True, seed=32)
  
train_generator = (pair for pair in zip(X, Y))
print("===========================")


model_checkpoint = ModelCheckpoint(f'model_checkpoints/class{config.CLASS_TYPE}' + 'unet5_{epoch:02d}.hdf5')


print("===========================")
print("TRAINING...")
model.fit_generator(train_generator, steps_per_epoch = len(img)/(16*2),  epochs=100, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint], validation_data=(x_val, y_val))
print("===========================")


print("===========================")
print("SAVING WEIGHTS...")
model.save_weights(f'weights/weights_class{config.CLASS_TYPE}.h5')
print("===========================")