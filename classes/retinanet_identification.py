import sys
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from classes.bounding_box import BoundingBox

class RetinaNet:
    model_path = None
    model = None
    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'Mass', 1: 'Calc'}

    def __init__(self, model_path):
        # adjust this to point to your downloaded/trained model
        # models can be downloaded here:
        # self.model_path = os.path.join('..', 'snapshots', model_path)
        # load retinanet model
        self.model = models.load_model(model_path, backbone_name='resnet50')


    def run_detection_path(self, image_path):
        # load image
        image = read_image_bgr(image_path)
        return self.run_detection(image)


    def run_detection(self, image):
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)
        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        return boxes, scores, labels, boxes/scale


    def draw_detection(self, image_path, boxes, scores, labels, threshold=0.5):
        image = read_image_bgr(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # visualize detections
        i = 1
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < threshold:
                break
            # if i in [1,2,4,6,7]:
            color = label_color(label)

            b = box.astype(int)
            draw_box(image, b, color=color)

            caption = "{} {:.3f}".format(self.labels_to_names[label], score)
            draw_caption(image, b, caption)
            # i += 1

        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(image)
        plt.show()


    def generate_bounding_boxes(self, boxes_real, scores, labels, threshold):
        boxes = []
        for box, score, label in zip(boxes_real[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < threshold:
                break
            box = box.astype(int)
            boxes.append(BoundingBox(box[0], box[1], box[2], box[3]))

        return boxes
        

if __name__ == '__main__':
    # use this to change which GPU to use
    gpu = "0"
    # set the modified tf session as backend in keras
    setup_gpu(gpu)
    work_dir = "/home/fundamentia/Descargas/"     
    diccom = "/home/fundamentia/python/corpus/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/Mass-Test_P_00470_RIGHT_CC/10-04-2016-DDSM-NA-59530/1.000000-full mammogram images-16388/1-1.dcm"
    img = "/home/fundamentia/python/corpus/transformadas_640/no_erosion/Mass-Test_P_00470_RIGHT_CC.png"

    # dic_img = img_pre.convert_diccom_to_png(diccom, work_dir)
    # dic_img = img_pre.crop_img(dic_img)
    # dic_img = img_pre.remove_noise(dic_img)
    # dic_img = img_pre.clahe(dic_img)

    # cv2.imwrite("/home/fundamentia/python/corpus/pruebas/diccimg.png", dic_img)
    # img = "/home/fundamentia/python/corpus/pruebas/diccimg.png"
    retinaNet = RetinaNet("/home/fundamentia/Descargas/resnet50_pascal_final.h5")
    boxes, scores, labels, boxes_real = retinaNet.run_detection_path(img)
    # retinaNet.cut_boxes("/home/fundamentia/python/corpus/pruebas/", img, boxes_real, scores, 0.1)
    retinaNet.draw_detection(img, boxes_real, scores, labels, 0.1)









