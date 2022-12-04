#importacion de librerias
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
import tensorflow_addons as tfa
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import os

gpus = tf.config.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



def create_inception(model_name, fold_path, model_path, optimizer=Adam(learning_rate=0.0001)):
  inputs = tf.keras.Input(shape=(224, 224, 3))
  head_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

  head_model.trainable = True

  head_model = head_model(inputs, training = True)
  head_model = tf.keras.layers.Flatten()(head_model)
  head_model = tf.keras.layers.Dense(256, activation='relu')(head_model)
  output = Dense(3, activation='softmax')(head_model)
  model4 = Model(inputs=inputs, outputs = output)

  train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
  )

  validation_datagen  = ImageDataGenerator(rescale=1./255)

  # Note that the validation data should not be augmented!
  train_generator = train_datagen.flow_from_directory(fold_path + '/Train',
                                                      batch_size=8,
                                                      class_mode='categorical',
                                                      target_size=(224, 224))     

  validation_generator =  validation_datagen.flow_from_directory(fold_path + '/Valid',
                                                          batch_size=8,
                                                          class_mode  = 'categorical',
                                                          target_size = (224, 224))

  # compilamos el modelo y lo entrenamos
  model4.compile(loss="categorical_crossentropy", 
                optimizer=optimizer,
                metrics=[tfa.metrics.F1Score(num_classes=3, average='micro'), 'accuracy'])
  
  return model4, train_generator, validation_generator


def train_densenet_model(model4, train_generator, validation_generator, model_path, model_name, epochs=100):
  batch_size = 8
  steps_per_epoch = train_generator.n // batch_size
  validation_steps = validation_generator.n // batch_size

  # generamos un monitor para el earlystop cuando el modelo este entrenado
  early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.001)
  # generamos el callback de guardado del modelo
  filepath = model_path + model_name + "_best.hdf5"
  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_f1_score', verbose=1, save_best_only=True, save_weights_only=True, mode='max')

  model4_history = model4.fit(
      train_generator,
      steps_per_epoch = steps_per_epoch,
      epochs = epochs,
      callbacks = [early_stop, checkpoint], 
      validation_data = validation_generator,
      validation_steps = validation_steps
  )
  return model4, model4_history


def train_k_fold_inception(kfold_path_original, kfold_models_path, k=5, epochs=100):
  # iteramos por cada fold para generar su modelo
  for i in range(k, 10):
    print("Entrenando fold {}".format(i))

    model_name = 'kfold_model_' + str(i)
    kfold_path = kfold_path_original + str(i)
    # generamos el modelo
    new_model, train_generator, validation_generator = create_inception(model_name, kfold_path, kfold_models_path, Adam(lr=0.0001))

    batch_size = 8
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    
    # generamos un monitor para el earlystop cuando el modelo este entrenado
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001)
    # generamos el callback de guardado del modelo
    filepath = kfold_models_path + model_name + "_best.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_f1_score', verbose=1, save_best_only=True, save_weights_only=True, mode='max')

    # entrenamos el model
    new_model_history = new_model.fit(
      train_generator,
      steps_per_epoch= steps_per_epoch,
      epochs= epochs,
      callbacks=[early_stop, checkpoint], 
      validation_data= validation_generator,
      validation_steps= validation_steps
    )
    # guardamos los resultados, history, best y last model
    np.save(kfold_models_path + '{}-History.npy'.format(model_name), new_model_history.history)
    new_model.save_weights(kfold_models_path + "{}_last.h5".format(model_name))


def evaluate_k_fold_inception(kfold_path_original, kfold_models_path, k=5):
  hist_all = []
  f1_all = []
  acc_all = []
  # iteramos por cada fold para generar su modelo
  for i in range(k, 10):
    # escogemos el modelo del k_fold
    print("Validando fold {}".format(i))
    model_name = 'kfold_model_' + str(i)
    kfold_path = kfold_path_original + str(i)

    # cargamos el demolo
    filepath = kfold_models_path + model_name + "_best.hdf5"
    model, train_generator, validation_generator = create_inception(model_name, kfold_path, kfold_models_path, Adagrad(lr=0.0001))
    model.load_weights(filepath)

    # cargamos el historial
    model_history = np.load(kfold_models_path + model_name + '-History.npy',allow_pickle='TRUE').item()
    hist_all.append(model_history)
    
    test_datagen  = ImageDataGenerator(rescale=1./255)
    test_generator =  test_datagen.flow_from_directory(kfold_path + '/Test',
            batch_size=8,
            class_mode  = 'categorical',
            target_size = (224, 224))

    test_lost, test_f1, test_acc = model.evaluate(test_generator)
    f1_all.append(test_f1)
    acc_all.append(test_acc)
    print("Fold {} -> F1 score: {} - Accuracy:{}".format(str(i), test_f1, test_acc))
    # test_generator.reset()
    # predict = model.predict(test_generator)
    # y_pred = np.rint(predict)
    # y_true = test_generator.classes
    # pred2 = []
    # for p in y_pred:
    #   pred2.append(np.argmax(p))

    # matrix = confusion_matrix(y_true, pred2)
    # fig, ax = plt.subplots(figsize=(5, 5))
    # disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['abnormal', 'normal'])
    # disp.plot(cmap=plt.cm.Blues, ax=ax)
    # plt.title("Matriz de confusión")
    # plt.show()


  # fold_labels = ['Fold0', 'Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5', 'Fold6', 'Fold7', 'Fold8', 'Fold9']
  # palette = sns.color_palette('hls', 10)
  # fig, ax = plt.subplots()
  # sns.set()
  # ax.set(ylim=(0,1))
  # ax.set(xlabel='Fold', ylabel='F1-score')
  # p = sns.barplot(x=fold_labels, y=f1_all, ax = ax, palette=palette)
  print("Media: {}".format(statistics.mean(f1_all)))
  print("Desviación standar: {}".format(statistics.stdev(f1_all)))


kfold_path = "/home/fundamentia/python/corpus/transformadas_640/clasificadas/Fold"
kfold_models_path = "/home/fundamentia/python/tfm_breast_cancer_detection/modelos/inception_models/kfold/"
# train_k_fold_inception(kfold_path, kfold_models_path, k=0, epochs=100)
evaluate_k_fold_inception(kfold_path, kfold_models_path, k=0)
