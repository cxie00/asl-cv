""" TODO: implement performance evaluation functions, call train_models, 
    use performance evaluation functions to test your results """
    
from train_sign_detector import trainSignDetector
from train_hand_detector import trainHandDetector
from sign_detector import recognize_gesture
import helpers
from sklearn.preprocessing import LabelEncoder



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
    #implement your code here
    return 0

def get_accuracy(test_list, signDetector, handDetector, label_encoder):
    #implement your code here
    return 0



label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])
    
user_list =['user_3', 'user_4','user_5','user_6','user_7','user_9','user_10']

trainlistsize = len(user_list)//2

train_list = user_list[:trainlistsize]

test_list = user_list[trainlistsize:]