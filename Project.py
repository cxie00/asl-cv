""" TODO: implement performance evaluation functions, call train_models, 
    use performance evaluation functions to test your results """
    
from train_sign_detector import trainSignDetector
from train_hand_detector import trainHandDetector
from sign_detector import recognize_gesture
import helpers
from sklearn.preprocessing import LabelEncoder
import numpy as np


def train_models(label_encoder, train_list):
    
    # train hand-not-hand classifier:
    # feel free to experiment with the hyperparameters here. The second parameter is the threshold at which training terminates 
    # (lower = more accurate but longer training time), the third parameter is maximum training iterations before it terminates
    # (higher = potentially more accurate, but potentially longer training time; this parameter only comes into play if threshold
    # is not met)
    handDetector = trainHandDetector(train_list, 40, 35)
    
    # save hand-not-hand classifier:
    helpers.dumpclassifier('handDetector.pkl', handDetector)
    
    # train sign classifier:
    signDetector = trainSignDetector(train_list, label_encoder)

    # save sign classifier:
    helpers.dumpclassifier('signDetector.pkl', signDetector)
    
    return handDetector, signDetector


# performance evaluation functions:

def get_confusion_matrix(test_list, signDetector, handDetector, label_encoder):
    #initialize values to 0
    matrix = np.zeros((24, 24))
    
    #get dictionary of form {filename: imagearray}
    imageset, _ = helpers.getHandSet(test_list, 'Dataset/')
    #incrementor to keep track of progress
    inc = 0
    #for each example in the test set
    for key in imageset:
        inc += 1
        print(inc)
        #get the predicted label
        outp = label_encoder.transform([recognize_gesture(imageset[key], handDetector, signDetector, label_encoder)[2]])[0]
        #get the actual label
        label = label_encoder.transform([key.split('/')[2][0]])[0]
        #increment the appropriate cell in the confusion matrix
        matrix[label][outp] += 1
    return matrix

def get_accuracy(test_list, signDetector, handDetector, label_encoder):
    #get dictionary of form {filename: imagearray}
    imageset, _ = helpers.getHandSet(test_list, 'Dataset/')
    #total number of examples in the test set
    total = len(imageset)
    #number of examples correctly classified
    correct = 0
    #incrementor to keep track of progress
    inc = 0
    #for each example in the test set
    for key in imageset:
        inc += 1
        print(inc)
        #get the predicted label
        outp = label_encoder.transform([recognize_gesture(imageset[key], handDetector, signDetector, label_encoder)[2]])
        #get the actual label
        label = label_encoder.transform([key.split('/')[2][0]])
        #if predicted and actual labels match, increment number correct
        if outp == label:
            correct += 1
    #accuracy is number correct / total number
    return correct/total

#label_encoder will encode letters to integers 0-23 (since j and z are excluded). Use label_encoder.transform([charlabel])[0] to get
#the corresponding number, and label_encoder.inverse_transform([num])[0] to get the corresponding character label
label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])
    
user_list =['user_3', 'user_4','user_5','user_6','user_7','user_9','user_10']

trainlistsize = 6 #how many users you want in your training set

train_list = user_list[:trainlistsize]

test_list = user_list[trainlistsize:]

#to retrain:
handDetector, signDetector = train_models(label_encoder, train_list)
accuracy = get_accuracy(test_list, signDetector, handDetector, label_encoder)

print(accuracy)

#to load previously trained models:
#handDetector = helpers.loadClassifier('handDetector.pkl')
#signDetector = helpers.loadClassifier('signDetector.pkl')
