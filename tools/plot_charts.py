import matplotlib.pyplot as plt
import csv
import pandas as pd


yolo_1 = yolo_2 = None
yolo_1 = pd.read_csv("/home/fundamentia/Dropbox/Master/Asignaturas/TFM/yolo/yolo1.txt", delimiter='\t', header=None)
yolo_2 = pd.read_csv("/home/fundamentia/Dropbox/Master/Asignaturas/TFM/yolo/yolo2.txt", delimiter='\t', header=None)




plt.plot(yolo_1[12])
plt.plot(yolo_2[12])
plt.title('Recall')
plt.ylabel('Recall')
plt.xlabel('epoch')
plt.legend(['defecto', 'custom'], loc='lower right')
plt.show()


plt.plot(yolo_1[11])
plt.plot(yolo_2[11])
plt.title('Precision')
plt.ylabel('Precision')
plt.xlabel('epoch')
plt.legend(['defecto', 'custom'], loc='lower right')
plt.show()