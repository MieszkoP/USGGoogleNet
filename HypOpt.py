import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import image
import matplotlib.pyplot as plt
import cv2
import skimage
from scipy import signal
path = "/Dataset_BUSI_with_GT" #Path
H = 400 #wysokosc
W = 400 #szerokosc

path_benign = path + "/benign"

L_benign = 437  # długość zbioru obrazów
h_benign = H  # wysokość obrazka
w_benign = W  # szerokość obrazka
L_designed = 500  # Docelowa ilość obrazków

labels_benign = np.hstack(((np.ones((L_designed, 1)), np.zeros((L_designed, 2)))))
labels_benign[:, [1, 0]] = labels_benign[:, [0, 1]]

path_malignant = path + "/malignant"

L_malignant = 210  # długość zbioru obrazów
h_malignant = H  # wysokość obrazka
w_malignant = W  # szerokość obrazka

labels_malignant = np.hstack(((np.ones((L_designed, 1)), np.zeros((L_designed, 2)))))
labels_malignant[:, [2, 0]] = labels_malignant[:, [0, 2]]

path_normal = path + "/normal"

L_normal = 133  # długość zbioru obrazów
h_normal = H  # wysokość obrazka
w_normal = W  # szerokość obrazka

labels_normal = np.hstack(((np.ones((L_designed, 1)), np.zeros((L_designed, 2)))))


def resizeAndNormalize(images, path, imageHeight, imageWidth, currentLength, type):
  for i in np.arange(0, currentLength):
    p = path + "/" + type + " (" + str(i + 1) + ").png"
    img = cv2.resize(image.imread(p)[:, :, 0], dsize=(400, 400), interpolation=cv2.INTER_CUBIC)  # Skalowanie do 400x400
    # print(np.mean(img))
    img = (img - np.mean(img)) / np.std(img)  # Normalizacja danych
    # print(i)
    if np.isnan(img).any() == True:
      print("ERROR")
      break
    images[:, :, i] = (img)


def augmentation(images, currentLength, designedLength):
  for i in np.arange(currentLength, designedLength):
    if i <= currentLength * 2:
      # img = cv2.GaussianBlur(image_benign[:,:,i-L_benign], (11,11),0)
      img = images[:, ::-1, i - currentLength]
    else:
      img = skimage.util.random_noise(images[:, :, i - 2 * currentLength], clip=False)
    # print(i)
    img = (img - np.mean(img)) / np.std(img)  # Normalizacja danych
    if np.isnan(img).any() == True:
      print("ERROR")
      break
    images[:, :, i] = (img)


def prepareData(path, imageHeight, imageWidth, currentLength, designedLength, type):
  images = np.zeros((imageHeight, imageWidth, designedLength))
  resizeAndNormalize(images, path, imageHeight, imageWidth, currentLength, type)
  augmentation(images, currentLength, designedLength)
  return images


image_benign = prepareData(path_benign, h_benign, w_benign, L_benign, L_designed, "benign")
image_malignant = prepareData(path_malignant, h_malignant, w_malignant, L_malignant, L_designed, "malignant")
image_normal = prepareData(path_normal, h_normal, w_normal, L_normal, L_designed, "normal")

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(image_normal[:, :, 7])
plt.axis("off")
plt.subplot(1, 2, 2)
plt.hist(image_normal[:, :, 7].ravel(), bins=256, range=(-1.5, 1.5), fc='k', ec='k')
plt.show()

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(image_normal[:, :, 133 + 7])
plt.axis("off")
plt.subplot(1, 2, 2)
plt.hist(image_normal[:, :, 133 + 7].ravel(), bins=256, range=(-1.5, 1.5), fc='k', ec='k')
plt.show()

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(image_normal[:, :, 266 + 7])
plt.axis("off")
plt.subplot(1, 2, 2)
plt.hist(image_normal[:, :, 266 + 7].ravel(), bins=256, range=(-1.5, 1.5), fc='k', ec='k')
plt.show()

##Formowanie aby można było je dać na wejście tensora:

print(image_malignant.shape)
image_normal = image_normal.swapaxes(0, 2)
image_benign = image_benign.swapaxes(0, 2)
image_malignant = image_malignant.swapaxes(0, 2)
labels = np.concatenate(([labels_normal, labels_benign, labels_malignant]))
images = np.concatenate(([image_normal, image_benign, image_malignant]))

data = list(zip(labels, images))
np.random.shuffle(data)
new_labels, new_images = zip(*data)
new_labels = np.asarray(new_labels)
new_images = np.asarray(new_images)

Length = new_images.shape[0]

train_ratio = 0.7

images_train = new_images[0:int(0.7 * Length), :, :]

images_test = new_images[int(0.7 * Length):, :, :]

labels_train = new_labels[0:int(0.7 * Length), :]

labels_test = new_labels[int(0.7 * Length):, :]

del Length, train_ratio, data, image_normal, image_benign, image_malignant, labels, images, new_images, new_labels

del model

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
from sklearn.metrics import classification_report, confusion_matrix

def inception(x,
              filters_1x1,
              filters_3x3_reduce,
              filters_3x3,
              filters_5x5_reduce,
              filters_5x5,
              filters_pool):
  path1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
  path2 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
  path2 = layers.Conv2D(filters_3x3, (1, 1), padding='same', activation='relu')(path2)
  path3 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
  path3 = layers.Conv2D(filters_5x5, (1, 1), padding='same', activation='relu')(path3)
  path4 = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  path4 = layers.Conv2D(filters_pool, (1, 1), padding='same', activation='relu')(path4)
  return tf.concat([path1, path2, path3, path4], axis=3)


inp = layers.Input(shape=(400, 400, 1))
x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inp)
x = layers.MaxPooling2D(3, strides=2)(x)
x = layers.Conv2D(64, 1, strides=1, padding='same', activation='relu')(x)
x = layers.Conv2D(192, 3, strides=2, padding='same', activation='relu')(x)
x = layers.MaxPooling2D(3, strides=2)(x)
x = inception(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32,
              filters_pool=32)
x = inception(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96,
              filters_pool=64)
x = layers.MaxPooling2D(3, strides=2)(x)
x = inception(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48,
              filters_pool=64)
aux1 = layers.AveragePooling2D((5, 5), strides=3)(x)
aux1 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux1)
aux1 = layers.Flatten()(aux1)
aux1 = layers.Dense(1024, activation='relu')(aux1)
aux1 = layers.Dropout(0.7)(aux1)
aux1 = layers.Dense(10, activation='softmax')(aux1)
x = inception(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64,
              filters_pool=64)
x = inception(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24, filters_5x5=64,
              filters_pool=64)
x = inception(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32, filters_5x5=64,
              filters_pool=64)
aux2 = layers.AveragePooling2D((5, 5), strides=3)(x)
aux2 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux2)
aux2 = layers.Flatten()(aux2)
aux2 = layers.Dense(1024, activation='relu')(aux2)
aux2 = layers.Dropout(0.7)(aux2)
aux2 = layers.Dense(10, activation='softmax')(aux2)
x = inception(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128,
              filters_pool=128)
x = layers.MaxPooling2D(3, strides=2)(x)
x = inception(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128,
              filters_pool=128)
x = inception(x, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48, filters_5x5=128,
              filters_pool=128)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(3, activation='softmax')(x)

from sklearn.metrics import multilabel_confusion_matrix

import pandas as pd

import seaborn as sn

losses = np.random.randint(-0.5,2,size=(7,1))        # 1
batch_sizes = np.random.randint(-0.5,4,size=(7,1))
solver_types = np.random.randint(-0.5,4,size=(7,1))
learning_rates = np.random.randint(-0.5,3,size=(7,1))
value = np.zeros((7,1))

generation = np.hstack((value, losses, batch_sizes, solver_types, learning_rates))

def ReadGenLoss(x):
  if x == 0:
    return tf.keras.losses.mean_squared_error
  elif x == 1:
    return tf.keras.losses.CategoricalCrossentropy()
  else:
    print("Error")

def ReadGenBatch(x):
  return int(20+20*x)

def ReadGenLR(x):
  if x == 0:
    return 0.001
  elif x == 1:
    return 0.0005
  elif x == 2:
    return 0.0001
  else:
    print("Error")

def ReadGenOpt(x):
  if x == 0:
    return tf.keras.optimizers.Adam
  elif x == 1:
    return tf.keras.optimizers.Adagrad
  elif x == 2:
    return tf.keras.optimizers.SGD
  elif x == 3:
    return tf.keras.optimizers.RMSprop
  else:
    print("Error")


for num_gen in range(0, 4):
  for j in range(0, 7):
    g = generation[j]
    print(g)
    fgust = np.zeros((3, 50))
    for i in range(0, 3):
      model = tf.keras.Model(inp, out, name="googlenet")
      model.compile(
        optimizer=ReadGenOpt(g[3])(ReadGenLR(g[4])),
        loss=ReadGenLoss(g[1]),
        metrics=['accuracy']
      )
      tf.keras.utils.plot_model(model, "googlenet.png", show_shapes=True)

      im_train = np.concatenate((images_train[:i * 350], images_train[i * 350 + 350:]))
      l_train = np.concatenate((labels_train[:i * 350], labels_train[i * 350 + 350:]))

      if i == 0 and j == 0 and num_gen == 0:
        model.save_weights('model.h5')
      else:
        model.load_weights('model.h5')

      history = model.fit(im_train, l_train, batch_size=ReadGenBatch(g[2]), epochs=50, validation_data=(
      images_train[0 + 350 * i:350 + 350 * i, :, :], labels_train[0 + 350 * i:350 + 350 * i, :]))

      predictions = model.predict(images_test)
      y_pred = np.argmax(predictions, axis=1)
      y_true = np.argmax(labels_test, axis=1)
      matrix = confusion_matrix(y_true, y_pred)

      plt.figure()
      matrix = confusion_matrix(y_true, y_pred)
      matrix = (matrix.T / (matrix.sum(axis=1))).T
      df_cm = pd.DataFrame(matrix, ['Zdrowy', 'Łagodny', 'Złosliwy'], ['Zdrowy', 'Łagodny', 'Złosliwy'])
      sn.set(font_scale=1.4)
      sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
      plt.show()

      b, a = signal.ellip(4, 0.01, 120, 0.125)
      fgust[i, :] = signal.filtfilt(b, a, history.history['val_accuracy'], method="gust")

      plt.figure()
      plt.plot(history.epoch, history.history['loss'], label="Strata zbioru treningowego", color='blue')
      plt.plot(history.epoch, history.history['val_loss'], label="Strata zbioru walidacyjnego", color="red")

      plt.figure()
      plt.plot(history.epoch, history.history['val_accuracy'], label="Metryka Accuracy", color='blue')
      plt.plot(history.epoch, fgust[i, :], 'r--')

      plt.legend(numpoints=1)
      plt.show()
      del model
      meaned_loss_value_this = fgust[i, :]
      maximum_acc_this = np.max(meaned_loss_value_this)
      if maximum_acc_this < 0.5:
        fgust[:, :] = np.vstack((fgust[i, :], fgust[i, :], fgust[i, :]))
        break

    meaned_loss_value = np.mean(fgust, axis=0)  # Policzenie średniej z accuracy po filracji
    maximum_acc = np.max(meaned_loss_value)  # Maksymalna wartość accuracy
    generation[j, 0] = maximum_acc
    print(generation[j])  # Wyznaczona generacja
  np.savetxt("generation(" + str(num_gen) + ").txt", generation)  # Zapisanie
  generation.sort(axis=0)  # Sortowanie

  # Krzyżowanie

  generation[0, 3:] = generation[4, 3:]
  generation[1, :3] = generation[4, :3]
  generation[1, 3:] = generation[5, 3:]
  generation[2, :3] = generation[5, :3]

  generation[2, 3:] = generation[6, 3:]
  generation[0, :3] = generation[6, :3]

  # Mutacja

  generation[np.random.randint(-0.5, 7), 1] = np.random.randint(0, 2)
  generation[np.random.randint(-0.5, 7), 2] = np.random.randint(0, 4)
  generation[np.random.randint(-0.5, 7), 3] = np.random.randint(0, 4)
  generation[np.random.randint(-0.5, 7), 4] = np.random.randint(0, 3)

  generation[np.random.randint(-0.5, 7), 1] = np.random.randint(0, 2)
  generation[np.random.randint(-0.5, 7), 2] = np.random.randint(0, 4)
  generation[np.random.randint(-0.5, 7), 3] = np.random.randint(0, 4)
  generation[np.random.randint(-0.5, 7), 4] = np.random.randint(0, 3)