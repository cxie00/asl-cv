""" Uses trained hand detector and sign detector to classify images """
""" TODO: implement sliding window approach """

from sklearn.preprocessing import LabelEncoder
from skimage.transform import rescale
import helpers
import matplotlib.pyplot as plt
import numpy as np


def sliding_window(model, img, stepSize, scale, min_scale = 0):
    """TODO: implement sliding window approach"""
    
    #dimensions of window
    windowWidth = 32
    windowHeight = 24
    
    #initialize list of best boxes
    boxes = []
    
    #we start with the unscaled image
    newimg = img
    newimgscale = 1
    
    # while scaled image is larger than window
    while len(newimg) > windowHeight and len(newimg[0]) > windowWidth and newimgscale > min_scale:  
        
        #initialize variables to find best box
        max_confidence = -1
        detected_box = []
        
        # slide a window across the image
        for y in range(0, len(newimg)-windowHeight, stepSize):
            for x in range(0, len(newimg[0])-windowWidth, stepSize):
                
                # yield the current window
                subimage = newimg[y:y + windowHeight, x:x + windowWidth]                
                # get probability that hand is contained in current window
                hogvec = helpers.convertToGrayToHOG(helpers.resize(subimage, ((128, 128))))
                confidence = model.predict_proba([hogvec])
                
                # if the current window is better than our best window so far, update best window
                if confidence[0][1] > max_confidence:
                    max_confidence = confidence[0][1]                    
                    detected_box = [x, y, max_confidence, newimgscale]
                    
        #get box coordinates in original image
        this_box = [0, 0, 0, 0, 0]
        this_box[0] = detected_box[0] // detected_box[3]
        this_box[1] = detected_box[1] // detected_box[3]
        this_box[2] = this_box[0] + (windowWidth // detected_box[3])
        this_box[3] = this_box[1] + (windowHeight // detected_box[3])
        this_box[4] = detected_box[2]
        
        #add best box for this scale to list of best boxes
        boxes.append(this_box)            
        #scale image
        newimgscale *= scale
        newimg = rescale(img, newimgscale)
        
    #perform NMS (don't worry too much about the details of this)
    box = helpers.non_max_suppression_fast(np.array(boxes), .4)
    
    return box[0]



def recognize_gesture(image, handDetector, signDetector, label_encoder):
    
        #find hand 
        #please play with the hyperparameters to find a balance between accuracy and runtime
        detected_box = sliding_window(handDetector, image, 50, .9)
        #crop image around hand
        croppedImage = helpers.crop(image, int(detected_box[0]), int(detected_box[2]), int(detected_box[1]), int(detected_box[3]))
        
        #get HOG of hand
        hogvec = helpers.convertToGrayToHOG(croppedImage)
        #use our trained model to get the probability of each class
        prediction_probs = signDetector.predict_proba([hogvec])[0]
        #get predicted class based on predicted probabilities
        prediction = label_encoder.inverse_transform(prediction_probs.tolist().index(max(prediction_probs)))

        return detected_box, prediction_probs, prediction
    
    
    
#handDetector = helpers.loadClassifier('handDetector.pkl')
#
#signDetector = helpers.loadClassifier('signDetector.pkl')
#
#label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
#       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])
#
#image = helpers.io.imread('Dataset/user_3/S2.jpg')
#
#loc, probs, pred = recognize_gesture(image, handDetector, signDetector, label_encoder)
#
#print(probs)