import sys, getopt, os
import cv2
import openface
import numpy as np
import math
import csv


def parseArgs(argv): # parse input arguments
    try:
        opts, args = getopt.getopt(argv,"i:")
        if not opts:
            print ('No options supplied')
            print ('Usage: RecognizeSinger.py -i <inputfolder>')
            sys.exit(2)
    except getopt.GetoptError as e:
        print ('Usage: RecognizeSinger.py -i <inputfolder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            folder = arg
    print ('Finding videos in folder %s ...' %folder)
    # make sure this is a folder
    if not os.path.isdir(folder):
        print ('Folder does not exist!')
        sys.exit(2)
    # create list of video files
    files = []
    for file in os.listdir(folder):
        if file.endswith('.mp4'):
            files.append(file)
    if not files:
        print ('No .mp4 files found!')
        sys.exit(2)
    if not folder.endswith('/'):
        folder = folder + '/'

    return files, folder

def getDuration(vidcap,ver):
    # numer of frames (version-specific)
    if int(ver[0])  < 3 :
        length = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    else :
        length = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    # frame-rate (version-specific)
    if int(ver[0])  < 3 :
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = vidcap.get(cv2.CAP_PROP_FPS)
    dur = float(length) / float(fps)
    return dur

def setPosition(vidcap,count,ver):
    if int(ver[0])  < 3 :
        vidcap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,count*1000)
    else:
        vidcap.set(cv2.CV_CAP_PROP_POS_MSEC,count*1000)

def mouthOpen(box,mouthTh,F): # check if mouth is open
    mdist = math.sqrt(math.pow(math.fabs(F[51][1]-F[57][1]),2)+math.pow(math.fabs(F[51][0]-F[57][0]),2))
    opening = mdist / float(box.bottom()-box.top())
    if opening > mouthTh:
        return True
    else:
        return False

def classify(rep,data,score,box,numClasses,k,lab): # k-NN classification
    reps = []
    reps.append((box.center().x, rep))
    r = reps[0]
    rep = r[1].reshape(1, -1)
    locScore = []
    for d in data:
        diff = np.linalg.norm(rep[0]-d)
        locScore.append(diff)
    scoreInds = [i[0] for i in sorted(enumerate(locScore), key=lambda x:x[1])]
    predClass = [0] * numClasses
    for i in range(0,k):
        predClass[int(lab[scoreInds[i]])] = predClass[int(lab[scoreInds[i]])] + float(k-i)/float(k)
    score = [a + b for a,b in zip(score,predClass)]
    return score

def main(argv):

    # parameters
    k = 5 # k-NN
    mouthTh = 0.15 # mouth-open threshold

    # check OpenCV version
    ver = (cv2.__version__).split('.')

    # load models, labels and embeddings
    dlibFacePredictor = './models/shape_predictor_68_face_landmarks.dat' # landmark prediction
    networkModel = "./models/nn4.small2.v1.t7" # face embedding

    # labels
    lab = []
    with open('./models/labels.csv','r') as c:
        myreader = csv.reader(c,delimiter=',')
        for row in myreader:
            lab.append(float(row[0]))
    numClasses = int(max(lab))+1
    labelMap = ['agujetas', 'antonioMairena', 'arcangel', 'argentina','camaron','chanoLobato','estrellaMorente','joseMenese','miguelPoveda','rocioMarquez']

    # embeddings
    data = []
    with open('./models/reps.csv','r') as d:
        myreader = csv.reader(d,delimiter=',')
        for row in myreader:
            line = []
            for el in row:
                line.append(float(el))
            data.append(line)

    # openface setup
    align = openface.AlignDlib(dlibFacePredictor) # face alignment
    net = openface.TorchNeuralNet(networkModel, imgDim=96, cuda=False) # face embedding

    # parse input arguments
    files, folder = parseArgs(argv)

    # iterate over files
    for file in files:
        print 'Processing file %s ...' %file
        score = [0] * numClasses
        vidcap = cv2.VideoCapture(folder + file) # read video
        dur = getDuration(vidcap, ver) # duration
        count = 0.0
        # frame-wise processing
        while count < dur:
            print (" %i / 100" %int(100*count / dur))
            setPosition(vidcap,count,ver) # roll video to current position

            success,image = vidcap.read() # read frame
            if not success:

                count = count + 1.0
                sys.stdout.write("\033[F")
                continue

            boxes = align.getAllFaceBoundingBoxes(image) # detect faces

            for box in boxes:
                F = align.findLandmarks(image,box) # estimate landmarks
                if mouthOpen(box,mouthTh,F): # check if mouth is open
                    # align face
                    alignedFace = align.align(96,image,box,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                    if alignedFace is None:
                        continue
                    # extract embedding
                    rep = net.forward(alignedFace)
                    # classify
                    score = classify(rep,data,score,box,numClasses,k,lab)

            count = count + 1.0
            sys.stdout.write("\033[F")

        # predict singer
        pred = score.index(max(score))
        person = labelMap[int(pred)-1]
        conf = float(score[pred]) / float(sum(score))
        print ("Predicted singer:'%s'with confidence %f" %(person,conf))


if __name__ == "__main__":
   main(sys.argv[1:])
