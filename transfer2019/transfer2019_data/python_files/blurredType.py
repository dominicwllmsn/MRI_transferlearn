import sys, subprocess, argparse, threading, time, queue, os, re

def getFilesinPathROI(path):
    '''
        Function to list all the files within a given path
    '''
    #Changing directory
    fileListStr = subprocess.check_output(["find", path,"-iname", '*gz'])
    fileListStr = fileListStr.decode()
    fileList = []
    for filePath in fileListStr.split('\n'):
        if filePath.strip():
            ind_stop = filePath.index('4_ROI_1mm') - 1
            fileList.append(filePath[:ind_stop].replace(path+'/', '')[11:])
    return fileList

def getFilesinPathNII(path):
    '''
        Function to list all the files within a given path
    '''
    #Changing directory
    fileListStr = subprocess.check_output(["find", path,"-iname", "*nii"])
    fileListStr = fileListStr.decode()
    fileList = []
    for filePath in fileListStr.split('\n'):
        if filePath.strip():
            #print(filePath)
            ind_ADNI = filePath.replace(path+'/', '').index('ADNI_')
            fileList.append(filePath.replace(path+'/', '')[ind_ADNI:])
            #print(filePath.replace(path+'/', '')[ind_ADNI:])
    return fileList

ROI_path = "./blurred_sub400"
nii_path = "./ADNI_comb"

ROIList = getFilesinPathROI(ROI_path)
niiList = getFilesinPathNII(nii_path)

#print(ROIList)
#print(niiList[0])
print('Number of ROI files:', len(ROIList))
print('Number of ADNI nii files:', len(niiList))

#print('S15147_I16392' in 'ADNI_002_S_0619_MR_MP-RAGE__br_raw_20060601215738863_1_S15147_I16392.nii')
with open('./blurred_sub400_type.txt', 'w+') as f_out:
    xfor line in ROIList:
        for i,nii in enumerate(niiList):
            if line+'.nii' in nii:
                print(nii)
                s_ind = nii[::-1].find('S')+1
                f_out.write(nii[-s_ind:-4]+'\n')
                break
           
# difference = list(set(niiList) - set(ROIList))
# print(len(difference))

# with open('./difference_comb.txt', 'w+') as f_out:
#            for i in difference:
#                 f_out.write(i+'\n')
