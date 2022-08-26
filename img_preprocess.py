import matplotlib.pyplot as plt
import pydicom as dicom
import os
# from google.colab.patches import cv2_imshow as cv2
import cv2 as cv2
import PIL # optional
import numpy as np
import png
import re
from skimage import util

#convertir diccom a png
def convert_diccom_to_png(in_file):
  ds = dicom.dcmread(os.path.join(in_file))
  pixel_array_numpy = ds.pixel_array
  return pixel_array_numpy


def show_diccom_img(diccom_path):
  ds = dicom.dcmread(diccom_path)
  plt.imshow(ds.pixel_array)
  plt.show()


def crop_img(img, x = 1, y = 4):
  # calculamos el % que se desea borrar
  y_start = int(img.shape[0]/(100-y))
  y_end = img.shape[0] - y_start
  x_start = int(img.shape[1]/(100-x))
  x_end = img.shape[1] - x_start
  print(y_start, y_end, x_start, x_end)
  # recortamos la imagen
  crop_img = img[y_start:y_end, x_start:x_end]
  return crop_img


# reescalar imagen con black background a 1024x1024
def rescale_img(img_object):
  # creamos una imagen negra de 1024x1024
  img_black = np.zeros((1024, 1024, 3), dtype="uint8")
  # leemos las dos imagenses y las reescalamos
  img_object = cv2.resize(img_object, (768, 1024))
  # posicionamos la imagen en la parte izquierda de la imagen negra
  x = 0
  y = 0
  x_end = x + img_object.shape[0]
  y_end = y + img_object.shape[1]
  img_black[x:x_end,y:y_end] = img_object
  return img_black


def remove_noise(img):
  # normalizamos la imagen
  norm = util.img_as_ubyte((img - img.min()) / (img.max() - img.min()))
  gray = cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY)

  # generamos los cortornos de la imagen
  contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  # escogemos el area de mayor tamaño
  max_area = max(contours, key=cv2.contourArea)
  
  # generamos una mascara con el area de mayor tamaño
  mask = np.zeros_like(img)
  cv2.drawContours(mask, [max_area], 0, (255, 255, 255), -1)

  # eliminamos el ruido de la imagen
  breast_img = cv2.bitwise_and(img, mask)
  return breast_img


# CLAHE (contrast limited adaptive histogram equalization)
def clahe(img, clip=2.0, tile=(5,5)):
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  lab_planes = list(cv2.split(lab))
  clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
  lab_planes[0] = clahe.apply(lab_planes[0])
  lab_planes[1] = clahe.apply(lab_planes[1])
  lab_planes[2] = clahe.apply(lab_planes[2])
  lab = cv2.merge(lab_planes)
  img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
  return img


#morphological erosion
def morphological_erosion(img, kernel_size=5):
  kernel = np.ones((kernel_size,kernel_size),np.uint8)
  erosion = cv2.erode(img,kernel,iterations = 1)
  return erosion


def process_ddsm_folder(input_folder, output_folder):
  for dir_path, dir_names, file_names in os.walk(input_folder, topdown=False):
    i = 1
    for file_name in file_names:
      if file_name.endswith(".dcm"):
        is_mask = "mask" in file_name if True else False
        #  obtenemos el nombre de la carpeta original
        folder_name = os.path.split(dir_path)[-2]
        # convertir diccom a png
        img = convert_diccom_to_png(os.path.join(dir_path, file_name))
        # recortar imagen
        img = crop_img(img)
        # reescalar imagen
        img = rescale_img(img)
        if not is_mask:
          # eliminar ruido
          img = remove_noise(img)
          # CLAHE
          img = clahe(img)
          # erosionar
          img = morphological_erosion(img)
          # guardar imagen
          cv2.imwrite(os.path.join(output_folder + "mamografias", folder_name + ".png"), img)
        else:
          if os.path.exists(os.path.join(output_folder + "mascaras", folder_name + "_" + str(i) + ".png")):
            i +=1
          # guardar imagen en la 
          cv2.imwrite(os.path.join(output_folder + "mascaras", file_name + "i".png), img)
        print("Processed: " + file_name)



# input_path = "/home/fundamentia/python/corpus/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/"
input_path = "/home/fundamentia/python/corpus/pruebas/"
output_path = "/home/fundamentia/python/corpus/transformadas/"
process_ddsm_folder(input_path, output_path)
