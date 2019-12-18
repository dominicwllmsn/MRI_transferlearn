import argparse, os
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import time

parser = argparse.ArgumentParser(description='Script for label generation, usage example: python3 label_generation.py /home/chao/data_3dcnn/data/label /home/chao/label.txt')
parser.add_argument('xmlLabelFolder', type=str, help='XML data folder. eg. /home/chao/data_3dcnn/data/label')
parser.add_argument('outputFile',  type=str, help='Output label file. eg. /home/chao/label.txt')

def path_exist(p):
    if not os.path.exists(p):
        return False
    return True

def get_labels( label_path):
    # collect labels start
    #label_path = '../data/label'
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

if __name__ == "__main__":
    opt = parser.parse_args()
    if not path_exist(opt.xmlLabelFolder):
        print('Folder or file does not exist!')
        exit()
    labels_image = get_labels(opt.xmlLabelFolder)
    np.savetxt(opt.outputFile, labels_image, fmt="%s")
    # np.savetxt("./label_all.txt", labels_image)
    # label = get_label()



def get_label( path='/home/chao/data_3dcnn/data/label_all.txt'):
    labels = np.loadtxt(path, dtype=np.str)
    label = {}
    for info in labels[:]:
        label[info[0]] = info[1]
    return label