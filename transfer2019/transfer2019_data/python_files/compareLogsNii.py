import sys, subprocess, argparse, threading, time, queue, os, re

def getFilesinPathLOG(path):
    '''
        Function to list all the files within a given path
    '''
    #Changing directory
    fileListStr = subprocess.check_output(["find", path,"-iname", '*txt'])
    fileListStr = fileListStr.decode()
    fileList = []
    for filePath in fileListStr.split('\n'):
        if filePath.strip():
            fileList.append(filePath.replace(path+'/', '')[14:-8]+'.nii')
    return fileList

def getFilesinPathNII(path):
    '''
        Function to list all the files within a given path
    '''
    #Changing directory
    fileListStr = subprocess.check_output(["find", path,"-iname", '*nii'])
    fileListStr = fileListStr.decode()
    fileList = []
    for filePath in fileListStr.split('\n'):
        if filePath.strip():
            fileList.append(filePath.replace(path+'/', ''))
    return fileList

logs_path = "./LogFiles_ALL"
nii_path = "./ADNI"

logsList = getFilesinPathLOG(logs_path)
niiList = getFilesinPathNII(nii_path)

#print(logsList[0])
#print(niiList[0])
print('Number of log files:', len(logsList))
print('Number of ADNI nii files:', len(niiList))

difference = list(set(niiList) - set(logsList))
print(len(difference))

with open('./difference_M.txt', 'w+') as f_out:
           for i in difference:
                f_out.write(i+'\n')
