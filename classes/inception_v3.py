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
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import os
import cv2


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
                                                      batch_size=12,
                                                      class_mode='categorical',
                                                      target_size=(224, 224))     

  validation_generator =  validation_datagen.flow_from_directory(fold_path + '/Valid',
                                                          batch_size=12,
                                                          class_mode  = 'categorical',
                                                          target_size = (224, 224))

  # compilamos el modelo y lo entrenamos
  model4.compile(loss="categorical_crossentropy", 
                optimizer=optimizer,
                metrics=[tfa.metrics.F1Score(num_classes=3, average='micro'), 'accuracy'])
  
  return model4, train_generator, validation_generator


def train_inception_model(model, train_generator, validation_generator, model_path, model_name, epochs=100):
  batch_size = 12
  steps_per_epoch = train_generator.n // batch_size
  validation_steps = validation_generator.n // batch_size

  # generamos un monitor para el earlystop cuando el modelo este entrenado
  early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001)
  # generamos el callback de guardado del modelo
  filepath = model_path + model_name + "_best.hdf5"
  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_f1_score', verbose=1, save_best_only=True, save_weights_only=True, mode='max')

  model4_history = model.fit(
      train_generator,
      steps_per_epoch = steps_per_epoch,
      epochs = epochs,
      callbacks = [early_stop, checkpoint], 
      validation_data = validation_generator,
      validation_steps = validation_steps
  )
  return model, model4_history


def evaluate_model(model, kfold_path):
  test_datagen  = ImageDataGenerator(rescale=1./255)
  test_generator =  test_datagen.flow_from_directory(kfold_path + '/Test',
          batch_size=16,
          class_mode  = 'categorical',
          target_size = (224, 224))

  test_lost, test_f1, test_acc = model.evaluate(test_generator)
  print('\n\n\nTEST ACCURACY:', test_acc)
  print('\n\n\n')
  return test_lost, test_f1, test_acc


# model, _, _ = create_inception("nombre", "/home/fundamentia/python/corpus/transformadas_640/clasificadas/Fold0", "", Adagrad(learning_rate=0.0001))
# model.load_weights("/home/fundamentia/python/tfm_breast_cancer_detection/modelos/inception_models/model_inception_hyper/model_DenseNet_Adagrad_0.0001_best.hdf5")

# image = cv2.imread("/home/fundamentia/python/corpus/transformadas_640/clasificadas/Fold0/Test/Calc/Calc-Test_P_00038_LEFT_MLO_1.png", cv2.COLOR_BGR2RGB)
# # images_list_norm = image.astype('float32')
# images_list_norm = image / 255.0

# model.predict(image)