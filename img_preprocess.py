import matplotlib.pyplot as plt
import pydicom as dicom
import os
from google.colab.patches import cv2_imshow as cv2
import PIL # optional

#convertir diccom a png
in_path = "/content/drive/MyDrive/TFM/Pruebas/originales/"
out_path = "/content/drive/MyDrive/TFM/Pruebas/procesadas/"
name = "man2.dcm"
out_name = name+ ".png"
def convert_diccom_to_png(in_file, out_path):
  ds = dicom.dcmread(os.path.join(in_path, name))
  pixel_array_numpy = ds.pixel_array
  cv2.imwrite(os.path.join(out_path, out_name), pixel_array_numpy)

image_path = in_path + "man2_mask2.dcm"
def show_diccom_img(diccom_path):
  ds = dicom.dcmread(image_path)
  plt.imshow(ds.pixel_array)
  plt.show()


convert_diccom_to_png(in_path + name, out_path + out_name)
show_diccom_img(in_path + name)