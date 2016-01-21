#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature= "salary"
poi  = "poi"
features_list = [poi, feature]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

import math
minValue = 0
maxValue = 0
for value in data:
    if not math.isnan(float(value[1])):
        if value[1] > maxValue:
            maxValue = value[1]
            if minValue == 0:
                minValue = value[1]
        elif value[1] < minValue:
            minValue = value[1]
        else:
            pass

print minValue, maxValue

