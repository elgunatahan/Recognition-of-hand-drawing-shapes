train.py
E

Tür:
Metin
Boyut
3 KB (3.426 bayt)
Kullanılan depolama alanı
3 KB (3.426 bayt)
Konum
SeniorProject
Sahibi
ben
Değiştirme:
8 Haz 2018; ben
Açma:
23:24; ben
8 Haz 2018 tarihinde Google Drive Web
ile oluşturuldu
Açıklama·ekleyin
Görüntüleyenler indirebilir

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

img_rows, img_cols = 32, 32

img_channels = 1 #Gray level

path1 = 'C:\\Users\\Elgun\\PycharmProjects\\SeniorProject\\trainData'  #the data set path should be given there

imlist = os.listdir(path1)
num_samples = size(imlist)
im1 = array(Image.open(path1 + '\\' + imlist[0]))

m, n = im1.shape[0:2]
imnbr = len(imlist)

immatrix = array([array(Image.open(path1 + '\\' + im2)).flatten()
                  for im2 in imlist], 'f')

label = np.ones((num_samples,), dtype=int)

label[0:1129] = 0
label[1129:1592] = 1
label[1592:3553] = 2
label[3553:6504] = 3
data, Label = shuffle(immatrix, label, random_state=2)
train_data = [data, Label]

batch_size = 64
nb_classes = 4
nb_epoch = 500
nb_filters = 32
nb_pool = 2
nb_conv = 3

(X, y) = (train_data[0], train_data[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0],img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(32, (3, 3),
                        border_mode='valid',
                        input_shape=(img_rows, img_cols,1), data_format='channels_last'))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(32, (3, 3)))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Convolution2D(32, (3, 3)))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                  verbose=1, validation_split=0.2)

model.save('my_model.h5')
model_json = model.to_json() #
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_weights.h5")
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])






