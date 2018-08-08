""" Sign Detection Creator """
"""TODO: implement performance evaluation functions, choose a model from sklearn """

"""Note that your sign detector model is trained on precropped images, so its performance is
    independent of the performance of the hand detection. Therefore, if this code has good 
    performance, but the overall program performs poorly, consider changing the hyperparameters 
    of your sliding window approach"""

import glob
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import helpers
import numpy as np
from sklearn.gaussian_process.kernels import RBF 
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#loads data for multiclass, returns two lists- X is a list of examples (HOGs), and Y is a list of corresponding correct labels
def get_data(user_list, img_dict, data_directory):
    X = []
    Y = []

    for user in user_list:
        user_images = glob.glob(data_directory+user+'/*.jpg')

        boundingbox_df = pd.read_csv(data_directory+user+'/'+user+'_loc.csv')
        
        for rows in boundingbox_df.iterrows():
            cropped_img = helpers.crop(img_dict[data_directory+rows[1]['image']], rows[1]['top_left_x'], rows[1]['bottom_right_x'], rows[1]['top_left_y'], rows[1]['bottom_right_y'])
            hogvector = helpers.convertToGrayToHOG(cropped_img)
            X.append(hogvector.tolist())
            Y.append(rows[1]['image'].split('/')[1][0])
    return X, Y


def trainSignDetector(train_list, label_encoder):
        """
            train_list : list of users to use for training
            eg ["user_1", "user_2", "user_3"]
        """
        imageset, _ = helpers.getHandSet(train_list, 'Dataset/')
        
        # Load data for the multiclass classification task
        X_mul,Y_mul = get_data(train_list, imageset, 'Dataset/')

        print("Multiclass data loaded")

        Y_mul = label_encoder.fit_transform(Y_mul)

        #Train Multiclass classifier using Sci-kit learn classifiers
        """TODO: Experiment with different models and hyperparameters to see what gives you the best results
        #For comparison of some options, see http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html """
        
        model = SVC(kernel="rbf", C=0.9, probability=True)
        # kernel='linear', C=0.9, probability=True
        
        signDetector = model.fit(X_mul, Y_mul)

        print("Sign Detector Trained")

        return signDetector
    
"""TODO: implement performance evaluation functions"""

def get_confusion_matrix(test_list, label_encoder, signDetector):
    #initialize all matrix values to 0
    matrix = np.zeros((24, 24))
    imageset, _ = helpers.getHandSet(test_list, 'Dataset/')
    #get test data
    X, Y = get_data(test_list, imageset, 'Dataset/')
    Y = label_encoder.transform(Y)
    #for each example in test data
    for i in range(len(X)):
        #get the model's prediction for that example
        outp = label_encoder.transform([helpers.predict(label_encoder, signDetector, X[i])])[0]
        #increment appropriate cell in matrix
        matrix[int(Y[i])][int(outp)] += 1
        
    return matrix

def get_accuracy(test_list, label_encoder, signDetector):
    #initialize correct counter to 0
    correct = 0
    imageset, _ = helpers.getHandSet(test_list, 'Dataset/')
    #get test data
    X, Y = get_data(test_list, imageset, 'Dataset/')     
    Y = label_encoder.transform(Y)
    #total is the number of examples in test data
    total = len(X)
    #for each example in test data
    for i in range(len(X)):
        #get the model's prediction for that example
        outp = label_encoder.transform([helpers.predict(label_encoder, signDetector, X[i])])[0]
        #if the prediction matched the correct label, increment the correct counter
        if Y[i] == outp:
            correct += 1
            
    return correct/total

label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])
    
user_list =['user_3', 'user_4','user_5','user_6','user_7','user_9','user_10']

trainlistsize = len(user_list)//2

train_list = user_list[:trainlistsize]
test_list = user_list[trainlistsize:]

signDetector = trainSignDetector(train_list, label_encoder)
matrix = get_confusion_matrix(test_list, label_encoder, signDetector)
accuracy = get_accuracy(test_list, label_encoder, signDetector)

print(matrix)
print(accuracy)