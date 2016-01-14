#!/usr/bin/python

import sys
import pickle
import pprint
import math
sys.path.append("../tools/")
import warnings
warnings.filterwarnings("ignore")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

"""
********************************************************************************
TASK 1 - Select what features you'll use
IMPORTANT NOTES: 
    Features selected using feature_selection.py
RESULT:
    'bonus', 'total_stock_value' & 'exercised_stock_options', are the default
    features selected plus X custom features named XXXXX 
    as explained in TASK 3
********************************************************************************
"""

features_list = [   'poi',
                    'bonus', 
                    'total_stock_value', 
                    'exercised_stock_options',
                    # 'salary_ratio',
                    # 'deferred_ratio',
                    # 'extras',
                    'salary_ratio_log',
                    'from_to_poi_ratio',
                    # 'restricted_stock_real'
                ]

"""
********************************************************************************
TASK 2 - Remove outliers
RESULT:
    There are two keys that seems to be clearly not real keys like "TOTAL" and
    "THE TRAVEL AGENCY IN THE PARK" that will be removed later in task 2.
********************************************************************************
"""
# for person in sorted(data_dict):
#     print person

# is_poi = 0
# total = 0
# for person in data_dict:   
#     if data_dict[person]['poi']:
#         is_poi += 1
#     total += 1
# print is_poi, total

for outlier in ['TOTAL','THE TRAVEL AGENCY IN THE PARK']:
    data_dict.pop(outlier,0)

"""
********************************************************************************
TASK 3 - Create new feature(s)
RESULT:
    
********************************************************************************
"""

### Check NaN
for person in data_dict:
    for feature in ['salary', 
                    'deferral_payments', 
                    'total_payments', 
                    'loan_advances', 
                    'bonus', #
                    'restricted_stock_deferred', 
                    'deferred_income', 
                    'total_stock_value', #
                    'expenses', 
                    'exercised_stock_options', #
                    'other', 
                    'long_term_incentive', 
                    'restricted_stock', 
                    'director_fees',
                    'to_messages', 
                    'from_poi_to_this_person', 
                    'from_messages', 
                    'from_this_person_to_poi', 
                    'shared_receipt_with_poi']:
        if math.isnan(float(data_dict[person][feature])):
            data_dict[person][feature] = 0.0
    # try:
    #     data_dict[person]['salary_ratio'] = float(data_dict[person]['salary']) / float(data_dict[person]['total_payments'])
    # except:
    #     data_dict[person]['salary_ratio'] = 0

    # try:
    #     data_dict[person]['deferred_ratio'] = float(abs(data_dict[person]['deferred_income'])) / float(data_dict[person]['total_payments'])
    # except:
    #     data_dict[person]['deferred_ratio'] = 0

    # try:
    #     data_dict[person]['extras'] = data_dict[person]['expenses'] + data_dict[person]['other'] + data_dict[person]['director_fees'] + data_dict[person]['long_term_incentive']
    # except: 
    #     data_dict[person]['extras'] = 0  

    try:
        data_dict[person]['salary_ratio_log'] = math.log(float(data_dict[person]['salary']) / float(data_dict[person]['total_payments']) + 1)
    except:
        data_dict[person]['salary_ratio_log'] = 0.0

    try:
        data_dict[person]['from_to_poi_ratio'] = float(data_dict[person]["from_poi_to_this_person"] + data_dict[person]["from_this_person_to_poi"] + data_dict[person]["shared_receipt_with_poi"]) / float(data_dict[person]['from_messages'] + data_dict[person]['to_messages'])
    except:
        data_dict[person]['from_to_poi_ratio'] = 0

    # try:
    #     data_dict[person]['restricted_stock_real'] = data_dict[person]['restricted_stock'] - data_dict[person]['restricted_stock_deferred']
    # except:
    #     data_dict[person]['restricted_stock_real'] = 0

### Store to my_dataset for easy export below.
my_dataset = data_dict
#pprint.pprint(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

"""
********************************************************************************
TASK 4 - Try a varity of clfs
IMPORTANT NOTES: 
    The data below, is extracted from feature_selection.py, form more
    information open that file in the same directory
RESULT:
    Naive Bayes, SVC & LinearSVC & Adaboost, discarded when using 
    feature_selection.py
********************************************************************************
"""

### NAIVE BAYES
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()

### DECISION TREE
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier()

### SUPPORT VECTOR MACHINE
# from sklearn.svm import SVC
# classifier = SVC()

### LINEAR SUPPORT VECTOR MACHINE
# from sklearn.svm import LinearSVC
# classifier = LinearSVC()

### RANDOM FOREST
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier()

### ADABOOST
# from sklearn.ensemble import AdaBoostClassifier
# classifier = AdaBoostClassifier()

"""
********************************************************************************
TASK 5.1 - Tuning the classifier
DESCRIPTION: 
    Using a pipeline with all the parameters to be tuned, the classifier, a 
    scorer and a CrossValidation method (StratifiedShuffleSplit) launch a 
    GridSearchCV that find best score value for a given scorer
RESULT:
    Surprisingly, best values are mostly the default values excepting the 
    splitter parameter set to 'random'
********************************************************************************
"""


# from sklearn.grid_search import GridSearchCV
# from sklearn.cross_validation import StratifiedShuffleSplit
# cv = StratifiedShuffleSplit(labels, n_iter = 10, random_state = 42)

# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()
## Parameters to use without selector
# parameters = {  
#                 'splitter':  ['best','random'],
#                 'criterion': ['entropy', 'gini'],
#                 'min_samples_split': [2, 4, 6, 8],
#                 'max_depth': [2, 4, 6],
#                 'class_weight': [None, "auto"],
#                 'max_leaf_nodes': [None] + range(2, 10, 1)
#              }

# from scorer import getScorings, printMetrics
## GridSearch witout selector
# clf = GridSearchCV( clf, 
#                     parameters,
#                     #n_jobs = -1, # all the CPUs of the computer. Don't use with getScorings
#                     cv = cv,
#                     verbose = 2,
#                     scoring = 'recall') # precision, recall, f1, getScorings

# clf.fit(features, labels)

# print '\nbest params\n'
# best_parameters = clf.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print '%s: %r' % (param_name, best_parameters[param_name])
# print 'Score:', clf.best_score_

## Print getScorings Metrics if getScoring is used as scorer, otherwise
## comment next line
#printMetrics(clf)

"""
********************************************************************************
TASK 5.2 - Tuned Classifier  
RESULT:
    Classifier calculations with tuned parameters. Print metrics table
********************************************************************************
"""

from helper import getMetrics
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(splitter="random")
getMetrics(clf, data, features_list)

"""
********************************************************************************
TASK 6 - Dump your classifier, dataset, and features_list
DESCRIPTION: 
    Store Project data to perfom testing
RESULT:
    my_classifier.pkl, my_dataset.pkl and my_feature_list.pkl
********************************************************************************
"""

dump_classifier_and_data(clf, my_dataset, features_list)

"""
********************************************************************************
ANNEX 0 - Code used while inspecting, testing and tunning
DESCRIPTION: 
    Pieces of code used in some stage of the project
********************************************************************************
"""

### Ploting features relations
# def Draw(mark_poi=False, f1_name="feature 1", f2_name="feature 2"):
#     import matplotlib.pyplot as plt
#     features_list_plot = ['poi', f1_name, f2_name]
#     data = featureFormat(my_dataset, features_list_plot, sort_keys = True)
#     poi, features = targetFeatureSplit( data )
#     idx = 0
#     for feature in features:
#         plt.scatter(feature[0], feature[1])
#     idx = 0
#     if mark_poi:
#         for feature in features:
#             if poi[idx]:
#                 plt.scatter(feature[0], feature[1], color="r", marker="*")
#             idx += 1
#     plt.xlabel(f1_name)
#     plt.ylabel(f2_name)
#     plt.show()

# Draw(mark_poi=True, f1_name="extras", f2_name="total_payments")


### Tunning RandomForest Classifier
# from sklearn.feature_selection import SelectKBest
# selector = SelectKBest()

# from sklearn.pipeline import Pipeline

### RANDOM FOREST
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()
## Parameters to use with selector
# parameters = {  'selector__k': [2, 3, 4, 5],
#                 'classifier__n_estimators': [50, 100, 150],
#                 # 'classifier__criterion': ['entropy', 'gini'],
#                 # 'classifier__min_samples_split': [2, 4, 6, 8],
#                 # 'classifier__max_depth': [2, 4, 6],
#                 # 'classifier__class_weight': [None, "auto"],
#                 # 'classifier__max_features': range(2, 3, 1) + ['auto', 'sqrt', 'log2'],
#                 # 'classifier__max_leaf_nodes': [None] + range(2, 10, 1)
#              }
## Parameters to use without selector
# parameters = {  
#                 'n_estimators': [50, 100, 150],
#                 'criterion': ['entropy', 'gini'],
#                 'min_samples_split': [2, 4, 6, 8],
#                 # 'max_depth': [2, 4, 6],
#                 # 'class_weight': [None, "auto"],
#                 # 'max_features': range(2, 3, 1) + ['auto', 'sqrt', 'log2'],
#                 # 'max_leaf_nodes': [None] + range(2, 10, 1)

### Tunning DecisionTree Classifier
### DECISION TREE
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()
## Parameters to use with selector
# parameters = {  'selector__k': [2, 3, 4, 5],
#                 'classifier__criterion': ['entropy', 'gini'],
#                 #'classifier__splitter':  ['best','random'],
#                 #'classifier__min_samples_split': [2, 4, 6, 8],
#                 #'classifier__max_depth': [2, 4, 6],
#                 #'classifier__class_weight': [None, "auto"],
#                 #'classifier__max_features': [None, 'auto', 'sqrt', 'log2'],
#                 #'classifier__max_leaf_nodes': range(2, 10, 1)
#              }

## GridSearch with selector
# pipeline = Pipeline([('selector', selector),('classifier', clf)])
# clf = GridSearchCV( pipeline, 
#                     parameters, 
#                     cv = cv,
#                     scoring = getScorings)

