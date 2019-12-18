'''
/**************************************************************************************
*    Title: adfinder.py
*    Author: Claudio A. Henriquez
*    Date: <2017>
*
***************************************************************************************/
'''
#Current command: python3 preprocess_F.py 6 /media/ADNI/Female/ADNI/ /media/ADNI/sys/AtlasROIMask/harvardoxford_prob_Hippo_Amyg_Thal_Puta_Pall_Caud_MTL_ROI_Mask_1mm_test.nii /media/ADNI/sys/output/F_ALL_out/

import sys, subprocess, argparse, threading, time, queue, os, re

parser = argparse.ArgumentParser(description='Script for ROI extraction, usage example: python3 preprocess.py 2 /media/chao/claudiolocal/ADNI_Screening/ /media/chao/claudiolocal/sys/AtlasROIMask/harvardoxford_prob_Hippo_Amyg_Thal_Puta_Pall_Caud_MTL_ROI_Mask_1mm_test.nii.gz /media/chao/claudiolocal/sys/output/o9/')
parser.add_argument('nthreads', type=int, help='Specify the number of thread used to process data')

parser.add_argument('fileFolderPath', type=str, help='Folder path including MRI data. eg. /media/chao/claudiolocal/ADNI_Screening/')
parser.add_argument('ROImaskFilename', type=str, help='MASK file path eg. /media/chao/claudiolocal/sys/AtlasROIMask/harvardoxford_prob_Hippo_Amyg_Thal_Puta_Pall_Caud_MTL_ROI_Mask_1mm_test.nii.gz')
parser.add_argument('OutputDir',  type=str, help='Output folder. eg. /media/chao/claudiolocal/sys/output/o9/')


def roiExtraction(target, inputFileName, ROIFileName, OutputFilename):
    target.write("fslmaths " + inputFileName + " -mul " + ROIFileName + " " + OutputFilename + "\n")
    target.write(str(subprocess.check_output(["fslmaths", inputFileName, "-mul", ROIFileName, OutputFilename])))


def check_path(p):
    if not os.path.exists(p):
        os.mkdir(p)

def fslthread(threadname, filepath):
    global ROBEXFolder, MNI152brain_1mm, brainFolder, affineFolder, segFolder, ROIFolder, logFolder, ROImaskFilename
    imageID = ''
    studyID = ''
    subjectID = ''
    outputFolder = ''

    if filepath == '':
        # Error, there is no file in the input
        raise ValueError('There is no input:', filepath, threadname)
    else:
        # p = re.compile('(\/?([a-zA-Z0-9_\.\-]+\/)*)(\w+\.nii)')
        p = re.compile('(\/?([a-zA-Z0-9_\.\-]+\/)*)((\w+|-)*\.nii)')
        m = p.match(filepath)
        if m == None:
            raise ValueError('Invalid Path:', filepath, threadname)
        else:
            path = m.group(1)
            filename = m.group(3)
            print(threadname + " >> Path:" + path)
            print(threadname + " >> Filename:" + filename)
            p2 = re.compile('ADNI_(\d+\_S\_\d+)(\w+|-)*\_S(\d+)\_I(\d+)\.nii')
            m2 = p2.match(filename)
            # Add condition for real files
            if m2 == None:
                subjectID = filepath
                studyID = filepath
                imageID = filepath
            else:
                subjectID = m2.group(1)
                studyID = m2.group(3)
                imageID = m2.group(4)
            print(threadname + " >> Subject ID:" + subjectID)
            print(threadname + " >> Study ID:" + studyID)
            print(threadname + " >> Image ID:" + imageID)

            # Output file names
            brainFilePath = brainFolder + subjectID + "_S" + studyID + "_I" + imageID + "_1_brain_1mm_test"
            affineFilePath = affineFolder + subjectID + "_S" + studyID + "_I" + imageID + "_2_brain_affine_1mm_test"

            segFilePath = segFolder + subjectID + "_S" + studyID + "_I" + imageID + "_2_brain_affine_seg_1mm_test"
            segGMFilePath = segFolder + subjectID + "_S" + studyID + "_I" + imageID + "_3_seg_GM_1mm_test"
            segWMFilePath = segFolder + subjectID + "_S" + studyID + "_I" + imageID + "_3_seg_WM_1mm_test"
            segCSFFilePath = segFolder + subjectID + "_S" + studyID + "_I" + imageID + "_3_seg_CSF_1mm_test"

            # ROI Files
            ROIFilePath = ROIFolder + subjectID + "_S" + studyID + "_I" + imageID + "_4_ROI_1mm_test"

            # Writing FSL execution output
            f = open(logFolder + 'ProcessOutput_' + filename.replace('.', '_') + '.txt', 'w')
            f.write("Process of image id: " + imageID + " has begun.\n")
            f.write("------------------------------------------------------\n")

            # Step 1: Brain extraction
            f.write("Step 1: BET \n")
            initialTime = time.time()
            startTime = initialTime
            f.write("Started at: " + time.ctime(startTime) + "\n")
            # Step 1: Command Execution

            # ROBEX
            f.write(
                ROBEXFolder + "runROBEX.sh" + " " + filepath + " " + brainFolder + subjectID + "_S" + studyID + "_I" + imageID + "_brain_test" + " " + "-R -f 0.5 -g 0" + "\n")
            f.write(str(subprocess.check_output(
                [ROBEXFolder + "runROBEX.sh", filepath, brainFilePath + ".nii", brainFilePath + "_mask.nii"])))
            # Compressing files
            f.write(str(subprocess.check_output(["gzip", brainFilePath + ".nii", brainFilePath + "_mask.nii"])))

            endTime = time.time()
            elapsedTime = endTime - startTime
            f.write("Finished at: " + time.ctime(endTime) + "\n")
            f.write("Elapsed time: " + time.strftime('%H:%M:%S', time.gmtime(elapsedTime)) + "\n")
            f.write("Step 1 has finished.\n")
            # Step 1: Brain extraction Finish
            f.write("------------------------------------------------------\n")

            # Step 2: FLIRT Affine
            f.write("Step 2: Affine Process \n")
            startTime = time.time()
            f.write("Started at: " + time.ctime(startTime) + "\n")
            # Step 2: Command Execution
            f.write(
                "flirt" + " -in " + brainFilePath + " -ref " + MNI152brain_1mm + " -out " + affineFilePath + ".nii.gz" + " -omat " + affineFilePath + ".mat" + " -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12" + "\n")
            f.write(str(subprocess.check_output(
                ["flirt", "-in", brainFilePath, "-ref", MNI152brain_1mm, "-out", affineFilePath + ".nii.gz", "-omat",
                 affineFilePath + ".mat", "-bins", "256", "-cost", "corratio", "-searchrx", "-90", "90", "-searchry",
                 "-90", "90", "-searchrz", "-90", "90", "-dof", "12"])))
            # "-interp", "trilinear"

            endTime = time.time()
            elapsedTime = endTime - startTime
            f.write("Finished at: " + time.ctime(endTime) + "\n")
            f.write("Elapsed time: " + time.strftime('%H:%M:%S', time.gmtime(elapsedTime)) + "\n")
            f.write("Step 2 has finished.\n")
            # Step 2: FLIRT Affine Finish
            f.write("------------------------------------------------------\n")

            # Step 3: FAST Segmentation
            f.write("Step 3: Segmentation Process \n")
            startTime = time.time()
            f.write("Started at: " + time.ctime(startTime) + "\n")
            # Step 3: Command Execution

            f.write("fast" + " -t 1 -n 3 -H 0.1 -I 8 -l 20.0 -B -b -o " + segFilePath + " " + affineFilePath + "\n")
            f.write(str(subprocess.check_output(
                ["fast", "-t", "1", "-n", "3", "-H", "0.1", "-I", "8", "-l", "20.0", "-B", "-b", "-o", segFilePath,
                 affineFilePath])))

            if os.path.isfile(segFilePath + "_pve_0.nii.gz") == True:
                f.write(str(subprocess.check_output(["mv", segFilePath + "_pve_0.nii.gz", segCSFFilePath + ".nii.gz"])))
                f.write(str(subprocess.check_output(["mv", segFilePath + "_pve_1.nii.gz", segGMFilePath + ".nii.gz"])))
                f.write(str(subprocess.check_output(["mv", segFilePath + "_pve_2.nii.gz", segWMFilePath + ".nii.gz"])))

            endTime = time.time()
            elapsedTime = endTime - startTime
            f.write("Finished at: " + time.ctime(endTime) + "\n")
            f.write("Elapsed time: " + time.strftime('%H:%M:%S', time.gmtime(elapsedTime)) + "\n")
            f.write("Step 3 has finished.\n")
            # Step 3: Segmentation Finished
            f.write("------------------------------------------------------\n")

            # Step 4: ROIs Extraction
            f.write("Step 4: ROIs Extraction\n")
            startTime = time.time()
            f.write("Started at: " + time.ctime(startTime) + "\n")
            # Step 4: Command Execution
            roiExtraction(f, segGMFilePath, ROImaskFilename, ROIFilePath)

            endTime = time.time()
            elapsedTime = endTime - startTime
            f.write("Finished at: " + time.ctime(endTime) + "\n")
            f.write("Elapsed time: " + time.strftime('%H:%M:%S', time.gmtime(elapsedTime)) + "\n")
            f.write("Step 4 has finished.\n")
            # Step 4: ROI Extraction Finished
            f.write("------------------------------------------------------\n")

#Defining class myThread
class myThread (threading. Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
    def run(self):
        print ("Starting " + self.name + "\n")
        process_data(self.name, self.q)
        print ("Exiting " + self.name + "\n")

# Queue Consumer
def process_data(threadName, q):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            data = q.get()
            queueLock.release()
            print ("%s processing %s \n" % (threadName, data))
            textrow = fslthread(threadName, data)

        else:
            queueLock.release()

def getFilesinPath(path):
    '''
        Function to list all the files within a given path
    '''
    #Changing directory
    fileListStr = subprocess.check_output(["find", "-name", '*.nii'], cwd=path)
    fileListStr = fileListStr.decode()
    fileList = []
    for filePath in fileListStr.split('\n')[76:500]:
        if filePath.strip():
            fileList.append( filePath.replace('./', path))
    return fileList


# Define a function for the thread
def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            threadName.exit()
        time.sleep(delay)
        print ("%s: %s\n" % (threadName, time.ctime(time.time())))
        counter -= 1


Test = False
# path of ROBEX
ROBEXFolder = "/media/ADNI/Toolkit/ROBEX/"
# path of MNI152 template
MNI152brain_1mm = "/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"

if __name__ == "__main__":
    opt = parser.parse_args()
    exitFlag = 0
    if len(sys.argv) == 5 or Test :
        if Test:
            nthreads = 2
            # folder path including MRI data
            fileFolderPath = '/media/ADNI/Female/ADNI/'
            # MASK file path
            ROImaskFilename = '/media/ADNI/sys/AtlasROIMask/harvardoxford_prob_Hippo_Amyg_Thal_Puta_Pall_Caud_MTL_ROI_Mask_1mm_test.nii.gz'
            # output folder path
            OutputDir = '/media/ADNI/sys/output/F_ALL_out/'
        else:
            nthreads = opt.nthreads
            fileFolderPath = opt.fileFolderPath
            ROImaskFilename = opt.ROImaskFilename
            OutputDir = opt.OutputDir

    else:
        print ("Usage:")
        print ("python3 preprocess.py <#Threads> <MRI Files Folder> <ROI Mask File> <Output Path>")
        print ("eg. python3 preprocess.py 2 /media/chao/claudiolocal/ADNI_Screening/ /media/chao/claudiolocal/sys/AtlasROIMask/harvardoxford_prob_Hippo_Amyg_Thal_Puta_Pall_Caud_MTL_ROI_Mask_1mm_test.nii.gz /media/chao/claudiolocal/sys/output/o9/")
        exit()
    if nthreads < 1 or not os.path.exists(fileFolderPath) or not os.path.exists(ROImaskFilename):
        print ("Wrong option")
        exit()

    # Folders and Files
    baseFolder = OutputDir
    outputFolder = baseFolder + 'OutputFiles/'
    brainFolder = outputFolder + '01_Brain/'
    affineFolder = outputFolder + '02_Affine/'
    segFolder = outputFolder + '03_Segmentation/'
    # nonLinearRegFolder = outputFolder + '04_NonLinearReg/'
    ROIFolder = outputFolder + '04_ROI/'
    logFolder = baseFolder + 'LogFiles/'
    check_path(baseFolder)
    check_path(outputFolder)
    check_path(brainFolder)
    check_path(affineFolder)
    check_path(segFolder)
    check_path(ROIFolder)
    check_path(logFolder)

    fileList = getFilesinPath(fileFolderPath)
    for index in range(len(fileList)):
        print(fileList[index])

    threadList = []
    for i in range(nthreads):
        threadList.append("Thread-{0:d}".format(i + 1))

    queueLock = threading.Lock()
    workQueue = queue.Queue(len(fileList) + 1)
    threads = []
    threadID = 1

    for tName in threadList:
        thread = myThread(threadID, tName, workQueue)
        thread.start()
        threads.append(thread)
        threadID += 1

    # Fill the queue
    queueLock.acquire()
    for ifile in fileList:
        workQueue.put(ifile)
    queueLock.release()

    # Wait for work queue to empty
    while not workQueue.empty():
        pass

    # Notify threads it's time to exit
    exitFlag = 1
