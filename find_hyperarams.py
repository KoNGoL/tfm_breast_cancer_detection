#importacion de librerias
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
import tensorflow_addons as tfa
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import os
import classes.densenet as densenet

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


optimizers = [{'type':Adam, 'name': 'Adam'}, {'type':SGD, 'name': 'SGD'}, {'type':Adagrad, 'name': 'Adagrad'}]
lrs = [0.01, 0.001, 0.0001]

model_path = '/home/fundamentia/python/tfm_breast_cancer_detection/modelos/DenseNet_models/model_DenseNet_hyper/'
folds_path = '/home/fundamentia/python/corpus/transformadas_640/clasificadas/'
i = 0
for opt in optimizers:
  for lr in lrs: 
    print(opt['name'])
    if  i < 8:
      i+=1
      continue
    model_name = "model_DenseNet_{}_{}".format(opt['name'], lr)
    model, train_generator, validation_generator = densenet.create_DenseNet(model_name, folds_path + 'Fold0', model_path, opt['type'](learning_rate=lr))
    model.summary()

    model, model4_history = densenet.train_DenseNet_model(model, train_generator, validation_generator, model_path, model_name, 40)
    np.save(model_path + '{}-History.npy'.format(model_name), model4_history.history)
