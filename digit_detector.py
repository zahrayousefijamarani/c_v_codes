import jupyter as jupyter
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.models import Sequential
import cv2

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape((-1, 28, 28, 1))
model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=4, kernel_size=(5, 5), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['acc'])
print(model.summary())

model.fit(x_train, y_train, epochs=1, batch_size=1)

test = x_test[11]
test = test.reshape(-1,28,28,1)
model.predict_classes(test)

img = cv2.imread("2.png")
img.reshape((-1,28,28,1))
model.predic_classes(img)

# plt.imshow(test)
# plt.imshow(x_train[20], cmp='gray')
# plt.title(y_train[20])
# plt.show()
