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
import random

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
def rescale_img(img_object, width, height):
  # creamos una imagen negra de 1024x1024
  img_black = np.zeros((width, height, 3), dtype="uint8")
  # calculamos el ratio de escalado
  # ratio = img_object.shape[0] / img_object.shape[1]
  # x = int(width / ratio)
  # leemos la imagen y la reescalamos
  img_object = cv2.resize(img_object, (width, height))
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
  max_area = max(contours, key=cv2.contourArea)
  rect = cv2.minAreaRect(max_area)
  # calculamos sus puntos
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  # print(box)
  max_x = [max(i) for i in zip(*box)][0]   
  min_x = [min(i) for i in zip(*box)][0] 
  max_y = [max(i) for i in zip(*box)][1] 
  min_y = [min(i) for i in zip(*box)][1] 
  return max_x, min_x, max_y, min_y

max_width = 0
max_height = 0

def create_pascal_vocal_xml(image_list, name, max_width, max_height):
  base_xml = '<annotation>\n\t<folder/>\n\t<filename>{name}</filename>\n\t<path/>\n\t<source>\n\t\t<database>https://www.cancerimagingarchive.net/</database>\n\t</source>\n\t<size>\n\t\t<width>640</width>\n\t\t<height>640</height>\n\t\t<depth>3</depth>\n\t</size>\n\t<segmented>0</segmented>\n\t{objects}\n</annotation>'
  base_objects_xml = '<object>\n\t\t<name>{type}</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n\t\t\t<xmin>{x_min}</xmin>\n\t\t\t<ymin>{y_min}</ymin>\n\t\t\t<xmax>{x_max}</xmax>\n\t\t\t<ymax>{y_max}</ymax>\n\t\t</bndbox>\n\t</object>'
  objects_xml = ''
  for image in image_list:
    x_max, x_min, y_max, y_min = detect_bounding_box(image)
    if (x_max - x_min) > max_width:
      max_width = x_max - x_min
    if (y_max - y_min) > max_height:
      max_height = y_max - y_min
    objects_xml = objects_xml + base_objects_xml.format(type = 'calc' if "Calc" in name else 'mass', x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max)

  xml = base_xml.format(name = re.sub("_\d.png", "", name), objects = objects_xml)
  return xml, max_width, max_height


def process_ddsm_folder(input_folder, output_folder):
  for dir_path, dir_names, file_names in os.walk(input_folder, topdown=False):
    for file_name in file_names:
      try:
        if file_name.endswith(".dcm"):
          if os.path.getsize(os.path.join(dir_path, file_name)) < 6534154:
            continue
          is_mask = "mask"  in dir_path if True else False
          if not is_mask:
            continue
          # if "1-1.dcm" in file_name and is_mask:
          #   continue
          #  obtenemos el nombre de la carpeta original
          # print(dir_path)
          folder_name = re.search(input_folder + "(.+?)(_\d+)?/.+", dir_path).group(1)
          # convertir diccom a png
          img = convert_diccom_to_png(os.path.join(dir_path, file_name))
          # cv2.imwrite(os.path.join(output_folder + "originales_mask/", folder_name + ".png"), img)
          # recortar imagen
          # img = crop_img(img)
          # # reescalar imagen
          # img = rescale_img(img, 640, 640)
          if not is_mask:
          #   # eliminar ruido
            img = remove_noise(img)
          #   # CLAHE
          #   img = clahe(img)
          #   cv2.imwrite(os.path.join(output_folder + "no_erosion/", folder_name + ".png"), img)
          #   # erosionar
          #   img = morphological_erosion(img)
          #   # guardar imagen
          #   cv2.imwrite(os.path.join(output_folder + "img/", folder_name + ".png"), img)
          else:
            i = 1
            while os.path.exists(os.path.join(output_folder + "originales_mask/", folder_name + "_" + str(i) + ".png")):
              i +=1
            # guardar imagen en la 
            cv2.imwrite(os.path.join(output_folder + "originales_mask", folder_name + "_" + str(i) + ".png"), img)
      except Exception as e:
        print("Processed: " + dir_path)


def generate_pascal_voc_xml(input_folder, output_folder):
  images_list = []
  last_name = ""
  max_width = 0
  max_height = 0
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
      xml, max_width, max_height = create_pascal_vocal_xml(images_list, last_name, max_width, max_height)
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
  xml, max_width, max_height = create_pascal_vocal_xml(images_list, last_name, max_width, max_height)
  # guardamos el xml
  with open(os.path.join(output_folder, last_name.replace(".png", ".xml")), "w") as xml_file:
    xml_file.write(xml)
  print("MAX WIDTH: " + str(max_width))
  print("MAX HEIGHT: " + str(max_height))


def cut_reescale_anomaly_img(input_folder_xml, input_folder_png, output_folder):
  for file_name in os.listdir(input_folder_xml):
    # leemos el XML
    # file_name = "Calc-Test_P_00077_RIGHT_MLO_1.xml"
    with open(input_folder_xml + file_name, 'r') as file:
     data = file.read()
    file.close()
    # obtenemos la imagen
    name = re.search('filename>(.+?)<', data).group(1)
    img = cv2.imread(os.path.join(input_folder_png, name + ".png"))
    try:
      # obtenemos los valores de los bounding boxes
      # aplicamos la regex y recorremos los matches
      matches = re.findall('<bndbox>[\s\S]+?xmin>(\d+)[\s\S]+?ymin>(\d+)[\s\S]+?xmax>(\d+)[\s\S]+?ymax>(\d+)[\s\S]+?</bndbox>', data)

      count = 1
      for match in matches:
        xmin = int(match[0])
        ymin = int(match[1])
        xmax = int(match[2])
        ymax = int(match[3])
        # ajustamos los valores para dar contexto a la img
        xmin = xmin - 100 if xmin - 100 > 0 else 0
        ymin = ymin - 100 if ymin - 100 > 0 else 0
        xmax = xmax + 100 if xmax + 100 < img.shape[1] else img.shape[1]
        ymax = ymax + 100 if ymax + 100 < img.shape[0] else img.shape[0]
        # recortamos la imagen
        img = img[ymin:ymax, xmin:xmax]
        # preprocesamos la imagen
        img = crop_img(img)
        # reescalar imagen
        img = rescale_img(img, 224, 224)
        # eliminar ruido
        img = remove_noise(img)
        # CLAHE
        img = clahe(img)
        #guardamos la imagen
        anomali_type = 'Calc'
        if "Mass" in name: 
          anomali_type = "Mass"
        cv2.imwrite(os.path.join(output_folder + "/" + anomali_type, name + "_" + str(count) + ".png"), img)
        count+=1
    except Exception as e:
      print(img.shape)
    


input_path = "/home/fundamentia/python/corpus/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/"
output_path = "/home/fundamentia/python/corpus/transformadas_640/"
# process_ddsm_folder(input_path, output_path)
# generate_pascal_voc_xml(output_path + "originales_mask/", output_path + "xml_separadas_originales/")

input_folder_xml = "/home/fundamentia/python/corpus/transformadas_640/xml_separadas_originales/"
input_folder_png = "/home/fundamentia/python/corpus/transformadas_640/originales/"
output_folder = "/home/fundamentia/python/corpus/transformadas_640/clasificadas/"
# cut_reescale_anomaly_img(input_folder_xml, input_folder_png, output_folder)

def create_k_fold(input_folder, output_folder):
  # leemos todas las imagenes
  for dir_name in os.listdir(input_folder):
    # leemos todas las imagenes de la carpeta
    images = []
    for file_name in os.listdir(input_folder + dir_name):
      images.append(file_name)
    # reodrdenamos la lista de forma aleatoria
    random.shuffle(images)
