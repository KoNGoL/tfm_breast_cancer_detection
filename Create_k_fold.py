import shutil
import matplotlib.pyplot as plt
import pydicom as dicom
import os
import cv2 as cv2
import PIL # optional
import numpy as np
import png
import re
from operator import itemgetter
from skimage import util
import random

def create_k_fold(input_folder, output_folder):
  # leemos todas las imagenes
  for dir_name in os.listdir(input_folder):
    # leemos todas las imagenes de la carpeta
    images = []
    for file_name in os.listdir(os.path.join(input_folder,dir_name)):
      images.append(file_name)
    # reodrdenamos la lista de forma aleatoria
    random.shuffle(images)
    # separamos los ficheros en lascarpetas de train, text y valid
    count = 0
    for img in images:
      # imagen para validacion
      folder_name = "Train"
      if count  == 7 or count  == 8:
        folder_name = "Valid"
      # imagen para test
      elif count == 9:
        folder_name = "Test"
        count = 0
      # copiamos la imagen a la carpeta correspondiente
      shutil.copy(os.path.join(input_folder, dir_name + "/" + img), os.path.join(output_folder, folder_name + "/" + dir_name + "/" + img))
      count+=1

kfold_input = "/home/fundamentia/python/corpus/transformadas_640/clasificadas/imagenes"
kfold_output= "/home/fundamentia/python/corpus/transformadas_640/clasificadas/Fold10"
create_k_fold(kfold_input,kfold_output)