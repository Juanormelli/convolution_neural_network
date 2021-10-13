import matplotlib.pyplot as plt

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout,normalization

from keras.utils import np_utils

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator



(X_train, y_train), (X_test, y_test)= mnist.load_data()
plt.imshow(X_train[1] )



inputs_train = X_train.reshape(X_train.shape[0],28,28,1)

inputs_test = X_test.reshape(X_test.shape[0],28,28,1)

inputs_train = inputs_train.astype("float32")

inputs_test = inputs_test.astype("float32")



inputs_train = inputs_train/255

inputs_test = inputs_test/255

outputs_train = np_utils.to_categorical(y_train)

outputs_test =np_utils.to_categorical(y_test)




classificator = Sequential()

classificator.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
classificator.add(BatchNormalization())
classificator.add(MaxPooling2D(pool_size=(2,2)))



classificator.add(Conv2D(32, (3,3), activation='relu'))
classificator.add(BatchNormalization())
classificator.add(MaxPooling2D(pool_size=(2,2)))
classificator.add(Flatten())




classificator.add(Dense(units=128, activation="relu"))
classificator.add(Dropout(0.2))
classificator.add(Dense(units=128, activation="relu"))
classificator.add(Dropout(0.2))
classificator.add(Dense(units=128, activation="relu"))

classificator.add(Dense(units=10, activation="softmax"))


classificator.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


generator_train = ImageDataGenerator(rotation_range=7,horizontal_flip=True, shear_range=0.2,height_shift_range=0.07, zoom_range=0.2)

generator_test = ImageDataGenerator()

base_train=generator_train.flow(inputs_train, outputs_train, batch_size=128)
base_test = generator_test.flow(inputs_test, outputs_test, batch_size=128)
 
classificator.fit_generator(base_train,steps_per_epoch=60000/128, epochs=5, validation_data=base_test,validation_steps=10000/128)



