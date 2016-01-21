import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

########################## DECISION TREE #################################



#### your code goes here
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

### be sure to compute the accuracy on the test set


    
def submitAccuracies():
  return {"acc":round(acc,3)}


print(submitAccuracies())
