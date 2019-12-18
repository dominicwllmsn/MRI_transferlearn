import os
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import time

def get_labels():
    # collect labels start
    label_path = '../data/label'
    file_list = os.listdir(label_path)
    labels_image = []
    for file in file_list:
        DOMTree = xml.dom.minidom.parse(label_path + "/" + file)
        collection = DOMTree.documentElement
        subjectIdentifiers = collection.getElementsByTagName("subjectIdentifier")[0]
        subjectIdentifier = subjectIdentifiers.childNodes[0].data
        subjectInfo = collection.getElementsByTagName("subjectInfo")[0]
        seriesIdentifiers = collection.getElementsByTagName("seriesIdentifier")[0]
        seriesIdentifier = "S" + seriesIdentifiers.childNodes[0].data
        imageUIDs = collection.getElementsByTagName("imageUID")[0]
        imageUID = "I" + imageUIDs.childNodes[0].data
        key = subjectIdentifier + "_" + seriesIdentifier + "_" + imageUID

        if subjectInfo.childNodes[0].data == 'AD':
            label = 2
        elif subjectInfo.childNodes[0].data == 'Normal':
            label = 0
        else:
            label = 1
        # info = tuple([key, str(label), subjectInfo.childNodes[0].data])
        labels_image.append([key, str(label), subjectInfo.childNodes[0].data])
    return labels_image


labels_image = get_labels()
np.savetxt("../data/label_all.txt", labels_image, fmt="%s")
# np.savetxt("./label_all.txt", labels_image)
# label = get_label()



def get_label( path='/home/chao/data_3dcnn/data/label_all.txt'):
    labels = np.loadtxt(path, dtype=np.str)
    label = {}
    for info in labels[:]:
        label[info[0]] = info[1]
    return label