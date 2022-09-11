import matplotlib.pyplot as plt
import pydicom as dicom
import os
# from google.colab.patches import cv2_imshow as cv2
import cv2 as cv2
import PIL # optional
import numpy as np
import png
import re
from operator import itemgetter
from skimage import util

work_dir = "/home/fundamentia/"

#convertir diccom a png
def convert_diccom_to_png(in_file):
  ds = dicom.dcmread(os.path.join(in_file))
  pixel_array_numpy = ds.pixel_array
  try:
    cv2.imwrite(os.path.join(work_dir + "borrar.png"), pixel_array_numpy)
    img = cv2.imread(os.path.join(work_dir + "borrar.png"))
  finally:
    # borramos la imagen tmp
    os.remove(os.path.join(work_dir + "borrar.png"))
  return img 


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
  # print(y_start, y_end, x_start, x_end)
  # print(y_end / x_end)
  # recortamos la imagen
  crop_img = img[y_start:y_end, x_start:x_end]
  return crop_img


# reescalar imagen con black background a 1024x1024
def rescale_img(img_object):
  # creamos una imagen negra de 1024x1024
  img_black = np.zeros((1024, 1024, 3), dtype="uint8")
  # calculamos el ratio de escalado
  ratio = img_object.shape[0] / img_object.shape[1]
  x = int(1024 / ratio)
  # leemos las dos imagenses y las reescalamos
  img_object = cv2.resize(img_object, (x, 1024))
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


def detect_bounding_box(img):
  #  cargamos la imagen
  # img = cv2.imread(image_path)
  # detectamos los contornos de la imagen
  norm = util.img_as_ubyte((img - img.min()) / (img.max() - img.min()))
  gray = cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY)
  contours,_ = cv2.findContours(gray, 1, 1) # not copying here will throw an error
  # detectamos el area de la masa
  rect = cv2.minAreaRect(contours[0])
  # calculamos sus puntos
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  # print(box)
  max_x = max(box, key=itemgetter(1))[0]  
  min_x = min(box, key=itemgetter(1))[0]
  max_y = max(box, key=itemgetter(0))[1]
  min_y = min(box, key=itemgetter(1))[1]
  return max_x, min_x, max_y, min_y


def create_pascal_vocal_xml(image_list, name):
  base_xml = '<annotation>\n\t<folder/>\n\t<filename>{name}</filename>\n\t<path/>\n\t<source>\n\t\t<database>https://www.cancerimagingarchive.net/</database>\n\t</source>\n\t<size>\n\t\t<width>1024</width>\n\t\t<height>1024</height>\n\t\t<depth>3</depth>\n\t</size>\n\t<segmented>0</segmented>\n\t{objects}\n</annotation>'
  base_objects_xml = '<object>\n\t\t<name>Mass</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n\t\t\t<xmin>{x_min}</xmin>\n\t\t\t<ymin>{y_min}</ymin>\n\t\t\t<xmax>{x_max}</xmax>\n\t\t\t<ymax>{y_max}</ymax>\n\t\t</bndbox>\n\t</object>'
  objects_xml = ''
  for image in image_list:
    x_min, x_max, y_min, y_max = detect_bounding_box(image)
    objects_xml = objects_xml + base_objects_xml.format(x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max)

  xml = base_xml.format(name = re.sub("_\d.png", "", name), objects = objects_xml)
  return xml


def process_ddsm_folder(input_folder, output_folder):
  for dir_path, dir_names, file_names in os.walk(input_folder, topdown=False):
    for file_name in file_names:
      try:
        if file_name.endswith(".dcm"):
          if os.path.getsize(os.path.join(dir_path, file_name)) < 6534154:
            continue
          is_mask = "mask" in dir_path if True else False
          if is_mask:
            continue
          # if "1-1.dcm" in file_name and is_mask:
          #   continue
          #  obtenemos el nombre de la carpeta original
          # print(dir_path)
          folder_name = re.search(input_folder + "(.+?)(_\d+)?/.+", dir_path).group(1)
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
            # img = morphological_erosion(img)
            # guardar imagen
            cv2.imwrite(os.path.join(output_folder + "no_erosion/", folder_name + ".png"), img)
          else:
            i = 1
            while os.path.exists(os.path.join(output_folder + "mascaras/", folder_name + "_" + str(i) + ".png")):
              i +=1
            # guardar imagen en la 
            cv2.imwrite(os.path.join(output_folder + "mascaras", folder_name + "_" + str(i) + ".png"), img)
      except Exception as e:
        print("Processed: " + dir_path)


def generate_pascal_voc_xml(input_folder, output_folder):
  images_list = []
  last_name = ""
  for file in sorted(os.listdir(input_folder)):
    # print(file)
    # if file.endswith("Mass-Training_P_01656_LEFT_CC_1.png"):
    #   file = file
    # si es la primera iteracion inseetamos el elemento en la lista
    if len(images_list) == 0:
      images_list.append(cv2.imread((os.path.join(input_folder, file))))
      last_name = file
    elif file.endswith("_1.png"):
      # si termina con _1, hacambiado la mamografia, generamos el xml y reiniciamos la lista
      xml = create_pascal_vocal_xml(images_list, last_name)
      # guardamos el xml
      with open(os.path.join(output_folder, last_name.replace(".png", ".xml")), "w") as xml_file:
        xml_file.write(xml)
      # reinicamos la lista 
      images_list = []
      images_list.append(cv2.imread((os.path.join(input_folder, file))))
      last_name = file
    else: 
      # si no termina con _1, seguimos con la misma mamografia e insertamos el elemento en la lista
      images_list.append(cv2.imread((os.path.join(input_folder, file))))
  
  # si la lista no esta vacia, generamos el xml
  xml = create_pascal_vocal_xml(images_list, last_name)
  # guardamos el xml
  with open(os.path.join(output_folder, last_name.replace(".png", ".xml")), "w") as xml_file:
    xml_file.write(xml)





# input_path = "/home/fundamentia/python/corpus/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/"
input_path = "/home/fundamentia/python/corpus/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/"
output_path = "/home/fundamentia/python/corpus/transformadas/"
# process_ddsm_folder(input_path, output_path)
generate_pascal_voc_xml(output_path + "mascaras/", output_path + "xml/")
