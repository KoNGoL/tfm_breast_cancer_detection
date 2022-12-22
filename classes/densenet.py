#importacion de librerias
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adagrad, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_densenet(fold_path, optimizer=Adam(learning_rate=0.0001), num_classes=2, batch_size=32):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    head_model = DenseNet121(weights = 'imagenet', include_top = False, input_shape = (224,224,num_classes))

    head_model.trainable = True

    head_model = head_model(inputs, training = True)
    head_model = tf.keras.layers.Flatten()(head_model)
    head_model = tf.keras.layers.Dense(256, activation='relu')(head_model)
    output = Dense(num_classes, activation='softmax')(head_model)
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

    # compilamos el modelo y lo entrenamos
    model4.compile(loss="categorical_crossentropy", 
                    optimizer=optimizer,
                    metrics=[tfa.metrics.F1Score(num_classes=num_classes, average='micro'), 'accuracy'])



    validation_datagen  = ImageDataGenerator(rescale=1./255)

    if (fold_path is None):
        return model4

    # Note that the validation data should not be augmented!
    train_generator = train_datagen.flow_from_directory(fold_path + '/Train',
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        target_size=(224, 224))     

    validation_generator =  validation_datagen.flow_from_directory(fold_path + '/Valid',
                                                            batch_size=batch_size,
                                                            class_mode  = 'categorical',
                                                            target_size = (224, 224))
    
    return model4, train_generator, validation_generator


def train_DenseNet_model(model, train_generator, validation_generator, model_path, model_name, epochs=100):
  batch_size = 8
  steps_per_epoch = train_generator.n // batch_size
  validation_steps = validation_generator.n // batch_size

  # generamos un monitor para el earlystop cuando el modelo este entrenado
  early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001)
  # generamos el callback de guardado del modelo
  filepath = model_path + model_name + "_best.hdf5"
  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')

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



# model, _, _ = create_DenseNet("nombre", "/home/fundamentia/python/corpus/transformadas_640/clasificadas/Fold0", "", Adagrad(learning_rate=0.0001))
# model.load_weights("/home/fundamentia/python/tfm_breast_cancer_detection/modelos/inception_models/model_inception_hyper/model_DenseNet_Adagrad_0.0001_best.hdf5")
# model = load_model("/home/fundamentia/python/tfm_breast_cancer_detection/modelos/inception_models/model_inception_hyper/model_DenseNet_Adagrad_0.0001_best.hdf5")

