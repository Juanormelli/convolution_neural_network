import matplotlib.pyplot as plt

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout

from keras.utils import np_utils

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import BatchNormalization

import numpy as np

from sklearn.model_selection import StratifiedKFold
seed =5 

np.random.seed(seed)

(X_train, y_train), (X_test, y_test)= mnist.load_data()
plt.imshow(X_train[1] )



inputs_train = X_train.reshape(X_train.shape[0],28,28,1)


inputs_train = inputs_train.astype("float32")




inputs_train = inputs_train/255



outputs_train = np_utils.to_categorical(y_train)

outputs_test =np_utils.to_categorical(y_test)


kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

results=[]

for index_train, index_test in kfold.split(inputs_train, np.zeros(shape=(inputs_train.shape[0],1))):
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
    classificator.fit(inputs_train[index_train],outputs_train[index_train],epochs=5,batch_size=100)
    
    precision = classificator.evaluate(inputs_train[index_test], outputs_train[index_test])
    results.append(precision[1])



mean = sum(results)/len(results)
