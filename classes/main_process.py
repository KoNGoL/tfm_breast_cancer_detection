import os
import sys
from pathlib import Path

sys.path.insert(1, '/home/fundamentia/python/tfm_breast_cancer_detection/')
sys.path.insert(2, '/home/fundamentia/python/tfm_breast_cancer_detection/yolov7/')

import cv2
import random
from shutil import rmtree
from classes.test_classifier import test_clasiffier_img_folder
import classes.img_preprocess_utils as img_preprocess_utils
from classes.retinanet_identification import RetinaNet
import classes.detect_custom as yolov7
from multiprocessing import Process, Queue

work_dir = None

def exec_diccom(diccom_image, masks_list, output_img, model_type='yolo', model_path=None):
    # realizamos el preprocesamiento
    img = pre_process_image(diccom_image)

    q = Queue()
    boxes = []
    if model_type == 'yolo':
        # ejecutamos yolo
        p = Process(target = exec_yolo, args=[img, q])
        p.start()
        boxes = q.get()
    else:
        # ejecutamos retinanet
        p = Process(target = exec_retinanet, args=[model_path, img, q])
        p.start()
        boxes = q.get()

    final_predictions = []
    if len(boxes) > 0:
        # cortamos los BB para generar las imagenes para clasificar
        cut_boxes(img, boxes)

        # ejecutamos la clasificacion
        p = Process(target = exec_classificacion, args=[q])
        p.start()
        final_predictions = q.get()
        print("Predictions: {}".format(final_predictions))

    others_count = generate_final_img(img, boxes, final_predictions, masks_list, output_img)
    return others_count


def pre_process_image(diccom_image_path):
    dic_img = img_preprocess_utils.convert_diccom_to_png(diccom_image_path, work_dir)
    dic_img = img_preprocess_utils.crop_img(dic_img)
    dic_img = img_preprocess_utils.remove_noise(dic_img)
    dic_img = img_preprocess_utils.rescale_img(dic_img, 640, 640, True)
    dic_img = img_preprocess_utils.clahe(dic_img)

    # cv2.imwrite("/home/fundamentia/tmp/final_img.png", dic_img)
    return dic_img


def exec_yolo(image, q):
    print("Ejecutando YOLO")
    # comprobamos que exista el direcctorio de trabajo de yolo
    yolo_work_dir = work_dir + "yolo/"
    if not os.path.exists(yolo_work_dir):
        os.makedirs(yolo_work_dir)

    #guardamos la imagen en la carpeta para analizar de yolo
    cv2.imwrite(yolo_work_dir + "/img.png", image)

    #ejecutamos yolo
    yolo_boxes = yolov7.main(source=yolo_work_dir)

    #eliminamos la imagen
    os.remove(yolo_work_dir + "/img.png")
    q.put(yolo_boxes)


def exec_retinanet(model_path, image, q):
    print("Ejecutando RETINANET")
    # comprobamos que exista el direcctorio de trabajo de yolo
    retinanet_work_dir = work_dir + "retinanet/"
    if not os.path.exists(retinanet_work_dir):
        os.makedirs(retinanet_work_dir)

    #guardamos la imagen en la carpeta para analizar de yolo
    cv2.imwrite(retinanet_work_dir + "/img.png", image)

    #ejecutamos yolo
    retinanet = RetinaNet(model_path)
    boxes, scores, labels, boxes_real = retinanet.run_detection(image)
    retinanet_boxes = retinanet.generate_bounding_boxes(boxes_real, scores, labels, 0.1)

    #eliminamos la imagen
    os.remove(retinanet_work_dir + "/img.png")
    q.put(retinanet_boxes)


def cut_boxes(image, bounding_boxes):
    # comprobamos que exista el direcctorio de trabajo para clasificacion
    class_work_dir = work_dir + "class/1/"
    Path(class_work_dir).mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(class_work_dir):
    #     os.makedirs(class_work_dir)

    i = 1
    for bb in bounding_boxes:
        coord = bb.get_x1(), bb.get_y1(), bb.get_x2(), bb.get_y2() 
        img_preprocess_utils.save_box(class_work_dir + "{}.png".format(i), image, coord)
        i += 1


def exec_classificacion(q):
    print("Ejecutando clasificacion")
    final_predictions = test_clasiffier_img_folder(models_path, work_dir + "class/")
    print(final_predictions)
    rmtree(work_dir + "class/")
    q.put(final_predictions)


def generate_final_img(image, bounding_boxes, final_predictions, mask_imgs = None, output_img= None):
    print("Generando imagen final")
    i = 0
    others_count = 0
    for bb in bounding_boxes:
        coord = bb.get_x1(), bb.get_y1(), bb.get_x2(), bb.get_y2()
        if final_predictions[i] == 2:
            i += 1
            others_count+=1
            continue
        tl = 1
        color = (0,0,255)
        cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), color, thickness=tl, lineType=cv2.LINE_AA)

        label = "Mass" if final_predictions[i] == 1 else "Calc"
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = coord[0] + t_size[0], coord[1] - t_size[1] - 3
        cv2.rectangle(image, (coord[0], coord[1]), c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (coord[0], coord[1]-2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        i += 1

    if mask_imgs is not None:
        image = mark_masks_in_image(image, mask_imgs)

    if output_img == None:
        output_img = work_dir + "final_img.png"
    cv2.imwrite(output_img, image)
    return others_count


def mark_masks_in_image(image, masks):
    for mask in masks:
        mask_image = img_preprocess_utils.convert_diccom_to_png(mask, work_dir)
        mask_image = img_preprocess_utils.crop_img(mask_image)
        mask_image = img_preprocess_utils.rescale_img(mask_image, 640, 640, True)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        # Aplica la máscara a la imagen original
        mask = cv2.bitwise_and(image, image, mask=mask_image)
        # # Aplica una máscara roja utilizando el método bitwise_or()
        image = cv2.addWeighted(image, 1.0, mask, 1.0, 0.0)

    return image


def exec_evaluate(path_to_evaluate):
    # leemos la carpeta
    current_folder_name = None
    mamografy_path = None
    mask_list = []
    continue_to = "Mass-Training_P_01650_LEFT_MLO"
    log = ""
    for folder_name in sorted(os.listdir(path_to_evaluate)):
        for dir_path, dir_names, file_names in os.walk(os.path.join(path_to_evaluate, folder_name), topdown=False):
            if current_folder_name == None or current_folder_name not in folder_name:
                if len(mask_list) > 0:
                    print(current_folder_name)
                    # ejecutamos la clasificacion
                    # if current_folder_name == "Calc-Test_P_00041_LEFT_CC":
                        # mam_image = img_preprocess_utils.convert_diccom_to_png(mamografy_path, work_dir)
                        # mam_image = img_preprocess_utils.crop_img(mam_image)
                        # try:
                        #     img = mark_masks_in_image(mam_image, mask_list)
                        #     cv2.imwrite("/home/fundamentia/python/corpus/evaluate_results/" + current_folder_name + ".png", img)
                        # except Exception as e:
                        #     print("Error al marcar las mascaras{}" + current_folder_name)
                        #     print(e)
                    
                    if continue_to is None:
                        others_count_retinanet = exec_diccom(mamografy_path, mask_list, output_path + "_retinanet/" + current_folder_name + ".png", 'retinanet', retinanet_model_path)
                        others_count_yolo = exec_diccom(mamografy_path, mask_list, output_path + "/" + current_folder_name + ".png", 'yolo')
                        log += "retinanet\t{}\t{}\n".format(current_folder_name, others_count_retinanet)
                        log += "yolo\t{}\t{}\n".format(current_folder_name, others_count_yolo)
                    elif current_folder_name == continue_to:
                        continue_to = None

                # escribirmos en fichero el string
                with open("/home/fundamentia/python/others.txt", "a") as f:
                    f.write(log)
                    log = ""

                current_folder_name = folder_name
                mamografy_path = dir_path + "/" + file_names[0]
                mask_list = []
                break
            else:
                for file_name in file_names:
                    if file_name.endswith(".dcm"):
                        if os.path.getsize(os.path.join(dir_path, file_name)) < 6534154:
                            continue
                        mask_list.append(os.path.join(dir_path, file_name))
                break

    # ejecutamos el ultimo fichero
    others_count_retinanet = exec_diccom(mamografy_path, mask_list, output_path + "_retinanet/" + current_folder_name + ".png", 'retinanet', retinanet_model_path)
    others_count_yolo = exec_diccom(mamografy_path, mask_list, output_path + "/" + current_folder_name + ".png", 'yolo')
    log += "retinanet\t{}\t{}\n".format(current_folder_name, others_count_retinanet)
    log += "yolo\t{}\t{}\n".format(current_folder_name, others_count_yolo)
    # escribirmos en fichero el string
    with open("/home/fundamentia/python/others.txt", "a") as f:
        f.write(log)
        log = ""

work_dir = "/home/fundamentia/tmp/"
models_path = '/home/fundamentia/python/tfm_breast_cancer_detection/modelos/'
output_path = "/home/fundamentia/python/corpus/evaluate_results"

if __name__ == '__main__':
    retinanet_model_path = "/home/fundamentia/Descargas/resnet50_pascal_final.h5"
    #cargamos el las dos imagenes
    diccom_image = "/home/fundamentia/python/corpus/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/Calc-Test_P_00041_LEFT_CC/08-29-2017-DDSM-NA-52275/1.000000-full mammogram images-92812/1-1.dcm"
    
    #ejecutar un fichero diccom
    # exec_diccom(diccom_image, None, None)

    #ejecutar un directorio de pngs para evaluar el resultado
    exec_evaluate("/home/fundamentia/python/corpus/evaluate/")

    



