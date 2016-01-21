#!/usr/bin/python
print "*******************"

"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

def getTruePositives(labels, predictions):
    cnt = 0
    truePositive = 0
    for prediction in predictions:
        if (labels[cnt] == 1 and prediction == 1):
            truePositive += 1
        cnt += 1
    print "The number of True Positives is : ", truePositive

def getTrueNegatives(labels, predictions):
    cnt = 0
    trueNegative = 0
    for prediction in predictions:
        if (labels[cnt] == 0 and prediction == 0):
            trueNegative += 1
        cnt += 1
    print "The number of True Negatives is : ", trueNegative

def getFalsePositives(labels, predictions):
    cnt = 0
    falsePositives = 0
    for prediction in predictions:
        if (labels[cnt] == 0 and prediction == 1):
            falsePositives += 1
        cnt += 1
    print "The number of False Positives is : ", falsePositives

def getFalseNegatives(labels, predictions):
    cnt = 0
    falseNegatives = 0
    for prediction in predictions:
        if (labels[cnt] == 1 and prediction == 0):
            falseNegatives += 1
        cnt += 1
    print "The number of False Positives is : ", falseNegatives

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print "Accuracy = ",accuracy_score(pred, labels_test)

# ****** Part 1 ******
# print sum(labels_test)

# ****** Part 2 ******
# print len(labels_test)

# ****** Part 3 ******
# fakePred = [0] * len(labels_test)
# print "fake accuracy = ",accuracy_score(fakePred, labels_test)

# ****** Part 4 ******
#getTruePositives(labels_test, pred)

# ****** Part 5 & 6 ******
# from sklearn.metrics import precision_score, recall_score
# print "Recall: ", recall_score(labels_test, pred)
# print "Precision: ", precision_score(labels_test, pred)

# ****** Part 7 ******
# from sklearn.metrics import precision_score, recall_score
# predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
# true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# print "Recall: ", recall_score(true_labels, predictions)
# print "Precision: ", precision_score(true_labels, predictions)
# getTruePositives(true_labels, predictions)

# ****** Part 8 ******
# predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
# true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# getTrueNegatives(true_labels, predictions)

# ****** Part 9 ******
# predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
# true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# getFalsePositives(true_labels, predictions)

# ****** Part 10 ******
# predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
# true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# getFalseNegatives(true_labels, predictions)

# ****** Part 11 ******
# predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
# true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# from sklearn.metrics import precision_score, recall_score
# print "Precision: ", precision_score(true_labels, predictions)

# ****** Part 12 ******
# predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
# true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# from sklearn.metrics import precision_score, recall_score
# print "Recall: ", recall_score(true_labels, predictions)
