#importacion de librerias
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import os
import cv2
import densenet as densenet
import inception_v3 as inception_v3
import vgg16 as vgg16
import img_preprocess_custom as img_preprocess

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


def test_clasiffier_img_folder(models_path, imgs_path): 
    test_datagen  = ImageDataGenerator(rescale=1./255)
    test_generator =  test_datagen.flow_from_directory(imgs_path,
            batch_size=16,
            class_mode  = 'categorical',
            target_size = (224, 224), shuffle=False)
    final_predictions = []
    # ejecutamos los modelos de vgg16
    model_vgg16 = vgg16.create_vgg16(None, Adagrad(learning_rate=0.0001), 3, 16)
    for i in range (0, 1):
        test_generator.reset()
        model_vgg16.load_weights(models_path + '/VGG16/' + 'kfold_model_{}_best.hdf5'.format(i))
        model_vgg16.compile(loss="categorical_crossentropy", 
                  optimizer=Adagrad(learning_rate=0.0001),
                  metrics=['accuracy'])
            
        final_predictions = exec_predicctions(final_predictions, model_vgg16, test_generator)

    # # ejecutamos los modelos de densenet
    # model_densenet = densenet.create_densenet(None, Adam(learning_rate=0.0001), 3, 16)
    # for i in range (0, 10):
    #     test_generator.reset()
    #     model_densenet.load_weights(models_path + '/DenseNet_models/kfold/' + 'kfold_model_{}_best.hdf5'.format(i))
    #     model_densenet.compile(loss="categorical_crossentropy", 
    #               optimizer=Adam(learning_rate=0.0001),
    #               metrics=['accuracy'])
            
    #     final_predictions = exec_predicctions(final_predictions, model_densenet, test_generator)

    # # ejecutamos los modelos de inception_v3
    # model_inception = inception_v3.create_inception(None, Adam(learning_rate=0.0001), 3, 16)
    # for i in range (0, 10):
    #     test_generator.reset()
    #     model_inception.load_weights(models_path + '/inception_models/kfold/' + 'kfold_model_{}_best.hdf5'.format(i))
    #     model_inception.compile(loss="categorical_crossentropy", 
    #               optimizer=Adam(learning_rate=0.0001),
    #               metrics=['accuracy'])
            
    #     final_predictions = exec_predicctions(final_predictions, model_inception, test_generator)

    # calculamos la preddiccion mas repetida
    for i in range (0, len(final_predictions)):
        final_predictions[i] = statistics.mode(final_predictions[i])
    
    # mostramos el resultado
    return final_predictions


def exec_predicctions(final_predictions, model, generator):
    prediction = model.predict(generator)
    # y_pred = np.rint(prediction)
    for j in range (0, len(prediction)):
        pred_val = np.argmax(prediction[j]) if prediction[j][np.argmax(prediction[j])] > 0.70 else 2
        if (len(final_predictions) == j):
            final_predictions.append([pred_val])
        else :
            final_predictions[j].append(pred_val)
    return final_predictions

models_path = '/home/fundamentia/python/tfm_breast_cancer_detection/modelos/'
imgs_path = "/home/fundamentia/python/corpus/transformadas_640/clasificadas/Fold{}/Test/"
imgs_path = "/home/fundamentia/python/corpus/pruebas/class/"

# for i in range (6, 10):
#     prediccionts = test_img_folder(models_path, imgs_path.format(i))
#     print("RESULTADO FOLD {}".format(i))
#     print(prediccionts)
final_predictions = test_clasiffier_img_folder(models_path, imgs_path)
print(final_predictions)