import flask 
from flask import after_this_request
import os
import time
from werkzeug.utils import secure_filename
import sys
import cv2
import base64
sys.path.insert(1, '/home/fundamentia/python/tfm_breast_cancer_detection/')
sys.path.insert(1, '/home/fundamentia/python/tfm_breast_cancer_detection/classes/')
import classes.main_process as mp
import classes.densenet as densenet

app = flask.Flask(__name__)

tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_dir")
models_path = '/home/fundamentia/python/tfm_breast_cancer_detection/modelos/'

@app.route("/")
def hello_world():
    return flask.render_template('index.html')


@app.route("/process_diccom", methods=["POST"])
def process_diccom():
    print("process_diccom")

    # guardamos el fichero subido
    diccom_file = flask.request.files["file"]

    # creamos un nombre de fichero único
    filename = secure_filename(diccom_file.filename)
    filename = str(time.time()) + "_" + filename
    file_path = os.path.join(tmp_dir, filename)

    # guardamos el fichero en el directorio temporal
    diccom_file.save(file_path)

    # marcamos el fichero para que se borre al finalizar la petición
    @after_this_request
    def remove_file(response):
        try:
            os.remove(file_path)
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response

    mp.work_dir = tmp_dir
    retinanet_out = os.path.join(tmp_dir, filename + "_retinanet_out.png")
    mp.exec_diccom(file_path, None, retinanet_out, 'retinanet', models_path + 'resnet50_pascal_final.h5')

    yolo_out = os.path.join(tmp_dir, filename + "_yolo_out.png")
    mp.exec_diccom(file_path, None, yolo_out)
    print(tmp_dir + filename + "_out.png")


    f_retinanet = open(retinanet_out, 'rb')
    base64_retinanet = base64.b64encode(f_retinanet.read()).decode("utf8")

    f_yolo = open(yolo_out, 'rb')
    base64_yolo = base64.b64encode(f_yolo.read()).decode("utf8")
    response = flask.make_response('{"retinanet":"' + base64_retinanet + '", "yolo":"' + base64_yolo + '"}')
    
    response.headers.set('Content-Type', 'text/json')
    response.headers.set('Content-Disposition', 'attachment', filename='retinanet_out.png')

    # marcamos el fichero para que se borre al finalizar la petición
    @after_this_request
    def remove_file(response):
        try:
            os.remove(retinanet_out)
            os.remove(yolo_out)
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response

    return response 


