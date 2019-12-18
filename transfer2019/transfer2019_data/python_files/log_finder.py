import sys, subprocess, argparse, threading, time, queue, os, re

def getFilesinPath(path):
    '''
        Function to list all the files within a given path
    '''
    #Changing directory
    fileListStr = subprocess.check_output(["find", ".","-iname", '*txt'])
    fileListStr = fileListStr.decode()
    fileList = []
    for filePath in fileListStr.split('\n'):
        if filePath.strip():
            fileList.append(filePath.replace('./', path))
    return fileList

logsList = getFilesinPath("./")

with open('./unfinished_logs_F.txt', 'w+') as f_out:
    for file_path in logsList:
        with open(file_path) as f_in:
            check = 0
            for line in f_in:
                if "step 4 has finished" in line.lower():
                    check = 1
            if check == 0:
                print(file_path[16:-8]+'.nii', "did not finish")
                f_out.write(file_path[16:-8]+'.nii'+'\n')
            