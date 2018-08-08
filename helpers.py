"""TODO: Feel free to implement any additional helper functions to use throughout your code here"""

from skimage import io
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
import pickle
import pandas as pd
import numpy as np




"""================== Please do not change anything below this line =================== """

#given a trained sign detector model, a feature vector, and a label_encoder, provides a prediction for the given feature vector 
def predict(label_encoder, signDetector, feature):
   prediction_probs = signDetector.predict_proba([feature])[0]
   prediction = label_encoder.inverse_transform(prediction_probs.tolist().index(max(prediction_probs)))
   return prediction

#given a list of filenames returns a dictionary of images 
def getfiles(filenames):
    dir_files = {}
    for x in filenames:
        dir_files[x]=io.imread(x)
    return dir_files

#return hog of a particular image vector
def convertToGrayToHOG(imgVector):
    rgbImage = rgb2gray(imgVector)
    return hog(rgbImage)

#takes image and coordinates, returns cropped image 
def crop(img,x1,x2,y1,y2):
    crp=img[y1:y2,x1:x2]
    crp=resize(crp,((128,128)))#resize
    return crp

#save classifier
def dumpclassifier(filename,model):
    with open(filename, 'wb') as fid:
        pickle.dump(model, fid)

#load classifier
def loadClassifier(picklefile):
    fd = open(picklefile, 'rb+')
    model = pickle.load(fd)
    fd.close()
    return model


#returns imageset and bounding box for a list of users 
def getHandSet(train_list, data_directory):
    frame = pd.DataFrame()
    list_ = []
    for user in train_list:
        list_.append(pd.read_csv(data_directory+user+'/'+user+'_loc.csv',index_col=None,header=0))
    frame = pd.concat(list_)
    frame['side']=frame['bottom_right_x']-frame['top_left_x']
    frame['hand']=1

    imageset = getfiles(('Dataset/'+frame.image.unique()))

    #returns actual images and dataframe 
    return imageset,frame


#performs NMS
def non_max_suppression_fast(boxes, overlapThresh):
    # print "Perfmoring NMS:"
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(s)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")