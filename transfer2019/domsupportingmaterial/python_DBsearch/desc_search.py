import sys, subprocess, argparse, threading, time, queue, os, re

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

nii_path = "./ADNI_comb"
niiList = getFilesinPathNII(nii_path)
desc_type = 'Field_Mapping'

#print(ROIList)
#print(niiList[0])
print('Number of ADNI nii files:', len(niiList))

count = 0
#print('S15147_I16392' in 'ADNI_002_S_0619_MR_MP-RAGE__br_raw_20060601215738863_1_S15147_I16392.nii')
with open('./'+desc_type+'_search.txt', 'w+') as f_out:
        for i,nii in enumerate(niiList):
            if desc_type in nii:
                count += 1
                if count%10:
                    print(count, 'done.')
                s_ind = nii[::-1].find('S')+1
                f_out.write(nii[-s_ind:-4]+'\n')
                continue
           
# difference = list(set(niiList) - set(ROIList))
# print(len(difference))

# with open('./difference_comb.txt', 'w+') as f_out:
#            for i in difference:
#                 f_out.write(i+'\n')
