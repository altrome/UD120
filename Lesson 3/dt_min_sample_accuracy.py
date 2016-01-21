import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################

from sklearn import tree
clf_2 = tree.DecisionTreeClassifier(min_samples_split=2)
clf_2.fit(features_train, labels_train)

clf_50 = tree.DecisionTreeClassifier(min_samples_split=50)
clf_50.fit(features_train, labels_train)

########################## DECISION TREE #################################



#### your code goes here
pred_2 = clf_2.predict(features_test)
pred_50 = clf_50.predict(features_test)
from sklearn.metrics import accuracy_score
acc_min_samples_split_2 = accuracy_score(pred_2, labels_test)
acc_min_samples_split_50 = accuracy_score(pred_50, labels_test)

### be sure to compute the accuracy on the test set


    
def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}


print(submitAccuracies())
