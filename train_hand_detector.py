""" Hand-not-Hand creator """
""" this code is complete and ready to use """


import random
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import helpers

#utility funtcion to compute area of overlap
def overlapping_area(detection_1, detection_2):
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)



#loads data for binary classification (hand/not-hand)
def load_binary_data(user_list, data_directory):
    data1,df = helpers.getHandSet(user_list, data_directory) # data 1 - actual images , df is actual bounding box
    
    # third return, i.e., z is a list of hog vecs, labels
    z = buildhandnothand_lis(df,data1)
    return data1,df,z[0],z[1]

#Creates dataset for hand-not-hand classifier to train on
#This function randomly generates bounding boxes 
#Return: hog vector of those cropped bounding boxes along with label 
#Label : 1 if hand ,0 otherwise 

def buildhandnothand_lis(frame,imgset):
    poslis =[]
    neglis =[]

    for nameimg in frame.image:
        tupl = frame[frame['image']==nameimg].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0
        
        dic = [0, 0]
        
        arg1 = [x_tl,y_tl,conf,side,side]
        poslis.append(helpers.convertToGrayToHOG(helpers.crop(imgset['Dataset/'+nameimg],x_tl,x_tl+side,y_tl,y_tl+side)))
        while dic[0] <= 1 or dic[1] < 1:
            x = random.randint(0,320-side)
            y = random.randint(0,240-side)
            crp = helpers.crop(imgset['Dataset/'+nameimg],x,x+side,y,y+side)
            hogv = helpers.convertToGrayToHOG(crp)
            arg2 = [x,y, conf, side, side]
            
            z = overlapping_area(arg1,arg2)
            if dic[0] <= 1 and z <= 0.5:
                neglis.append(hogv)
                dic[0] += 1
            if dic[0]== 1:
                break
    label_1 = [1 for i in range(0,len(poslis)) ]
    label_0 = [0 for i in range(0,len(neglis))]
    label_1.extend(label_0)
    poslis.extend(neglis)
    return poslis,label_1


# Does hard negative mining and returns list of hog vectos , label list and no_of_false_positives after sliding 

def do_hardNegativeMining(cached_window,frame, imgset, model, step_x, step_y):   
    lis = []
    no_of_false_positives = 0
    for nameimg in frame.image:
        tupl = frame[frame['image']==nameimg].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0
        
        dic = [0, 0]
        
        arg1 = [x_tl,y_tl,conf,side,side]
        for x in range(0,320-side,step_x):
            for y in range(0,240-side,step_y):
                arg2 = [x,y,conf,side,side]
                z = overlapping_area(arg1,arg2)
                
                
                prediction = model.predict([cached_window[str(nameimg)+str(x)+str(y)]])[0]

                if prediction == 1 and z<=0.5:
                    lis.append(cached_window[str(nameimg)+str(x)+str(y)])
                    no_of_false_positives += 1
    
    label = [0 for i in range(0,len(lis))]
    return lis,label, no_of_false_positives


# Modifying to cache image values before hand so as to not redo that again and again 
def cacheSteps(imgset, frame ,step_x,step_y):
    # print "Cache-ing steps"
    list_dic_of_hogs = []
    dic = {}
    i = 0
    for img in frame.image:
        tupl = frame[frame['image']==img].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0
        i += 1 
        if i%10 == 0:
             print(i, " images cached")
        image = imgset['Dataset/'+img]
        for x in range(0,320-side,step_x):
            for y in range(0,240-side,step_y):
                dic[str(img+str(x)+str(y))]=helpers.convertToGrayToHOG(helpers.crop(image,x,x+side,y,y+side))
    return dic    



def improve_Classifier_using_HNM(hog_list, label_list, frame, imgset, threshold=50, max_iterations=25): # frame - bounding boxes-df; yn_df - yes_or_no df
    # print "Performing HNM :"
    no_of_false_positives = 1000000     # Initialise to some random high value
    i = 0

    step_x = 32
    step_y = 24

    mnb  = MultinomialNB()
    cached_wind = cacheSteps(imgset, frame, step_x, step_y)

    while True:
        i += 1
        model = mnb.partial_fit(hog_list, label_list, classes = [0,1])

        ret = do_hardNegativeMining(cached_wind,frame, imgset, model, step_x=step_x, step_y=step_y)
        
        hog_list = ret[0]
        label_list = ret[1]
        no_of_false_positives = ret[2]
        
        if no_of_false_positives == 0:
            return model
        
        print("Iteration", i, "- No_of_false_positives:", no_of_false_positives) 
        
        if no_of_false_positives <= threshold:
            return model
        
        if i>max_iterations:
             return model
         
def trainHandDetector(train_list, threshold, max_iterations):
    imageset, boundbox, hog_list, label_list = load_binary_data(train_list, 'Dataset/')
    print('Binary data loaded')
    handDetector = improve_Classifier_using_HNM(hog_list, label_list, boundbox, imageset, threshold=threshold, max_iterations=max_iterations)
    print('Hand Detector Trained')
    return handDetector