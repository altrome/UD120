#!/usr/bin/python

import sys
import pickle
import pprint
from math import isnan
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

for outlier in ['TOTAL','THE TRAVEL AGENCY IN THE PARK']:
    data_dict.pop(outlier,0)

"""
********************************************************************************
TASK 3 - Create new feature(s)
RESULT:
    
********************************************************************************
"""

# is_poi = 0
# total = 0
# for person in data_dict:   
#     if data_dict[person]['poi']:
#         is_poi += 1
#     total += 1
# print is_poi, total

for person in data_dict:
    # NaN ---> 0
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
        if isnan(float(data_dict[person][feature])):
            data_dict[person][feature] = 0
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
        data_dict[person]['salary_ratio_log'] = Math.log(float(data_dict[person]['salary']) / float(data_dict[person]['total_payments']) + 1)
    except:
        data_dict[person]['salary_ratio_log'] = 0

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

# for person in my_dataset:
#     if my_dataset[person]['poi'] > 100000000:
#         print person
#     if my_dataset[person]['extras'] > 6000000:
#         print person


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
### ++++++++++++++++++ RESULTS +++++++++++++++++
### Scoring: Recall
###     Features selected: 19
###     Score: 0.8345
###     Execution time (1000 iterations): 109.1s
###     Metrics: 
###         Accur | Preci | Recll | F1_sc | features 
###         0.837 | 0.237 | 0.193 | 0.201 | 02 
###         0.841 | 0.321 | 0.268 | 0.275 | 03 
###         0.848 | 0.368 | 0.331 | 0.327 | 04 
###         0.848 | 0.379 | 0.359 | 0.347 | 05 
###         0.841 | 0.337 | 0.320 | 0.308 | 06 
###         0.837 | 0.314 | 0.297 | 0.287 | 07 
###         0.836 | 0.301 | 0.282 | 0.274 | 08 
###         0.839 | 0.322 | 0.291 | 0.288 | 09 
###         0.844 | 0.344 | 0.301 | 0.301 | 10 
###         0.832 | 0.348 | 0.322 | 0.307 | 11 
###         0.795 | 0.316 | 0.341 | 0.287 | 12 
###         0.772 | 0.307 | 0.343 | 0.284 | 13 
###         0.679 | 0.269 | 0.436 | 0.272 | 14 
###         0.449 | 0.192 | 0.714 | 0.259 | 15 
###         0.320 | 0.148 | 0.834 | 0.249 | 16 
###         0.301 | 0.141 | 0.825 | 0.239 | 17 
###         0.304 | 0.141 | 0.831 | 0.240 | 18 
###         0.335 | 0.147 | 0.834 | 0.250 | 19 <---

### DECISION TREE
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier()
### ++++++++++++++++++ RESULTS +++++++++++++++++
### Scoring: Recall
###     Features selected: 3
###     Score: 0.356
###     Execution time (1000 iterations): 110.5s
###     Metrics: 
###         Accur | Preci | Recll | F1_sc | features 
###         0.783 | 0.153 | 0.187 | 0.159 | 02 
###         0.811 | 0.299 | 0.354 | 0.305 | 03 <---
###         0.807 | 0.269 | 0.307 | 0.268 | 04 
###         0.804 | 0.249 | 0.279 | 0.246 | 05 
###         0.804 | 0.244 | 0.266 | 0.237 | 06 
###         0.804 | 0.234 | 0.268 | 0.234 | 07 
###         0.809 | 0.251 | 0.286 | 0.250 | 08 
###         0.808 | 0.259 | 0.285 | 0.254 | 09 
###         0.806 | 0.241 | 0.271 | 0.239 | 10 
###         0.803 | 0.225 | 0.250 | 0.222 | 11 
###         0.799 | 0.215 | 0.240 | 0.213 | 12 
###         0.797 | 0.204 | 0.234 | 0.205 | 13 
###         0.797 | 0.202 | 0.229 | 0.201 | 14 
###         0.794 | 0.194 | 0.227 | 0.196 | 15 
###         0.796 | 0.195 | 0.222 | 0.193 | 16 
###         0.795 | 0.203 | 0.226 | 0.199 | 17 
###         0.797 | 0.206 | 0.226 | 0.201 | 18 
###         0.796 | 0.199 | 0.223 | 0.197 | 19 

### SUPPORT VECTOR MACHINE
# from sklearn.svm import SVC
# classifier = SVC()
### ++++++++++++++++++ RESULTS +++++++++++++++++
### Scoring: Recall
###     Features selected: 2
###     Score: 0
###     Execution time (1000 iterations): 147.4s
###     Metrics: 
###         Accur | Preci | Recll | F1_sc | features
###         0.857 | 0.000 | 0.000 | 0.000 | 02 <---
###         0.867 | 0.000 | 0.000 | 0.000 | 03 
###         0.867 | 0.000 | 0.000 | 0.000 | 04 
###         0.867 | 0.000 | 0.000 | 0.000 | 05 
###         0.867 | 0.000 | 0.000 | 0.000 | 06 
###         0.867 | 0.000 | 0.000 | 0.000 | 07 
###         0.867 | 0.000 | 0.000 | 0.000 | 08 
###         0.867 | 0.000 | 0.000 | 0.000 | 09 
###         0.867 | 0.000 | 0.000 | 0.000 | 10 
###         0.867 | 0.000 | 0.000 | 0.000 | 11 
###         0.867 | 0.000 | 0.000 | 0.000 | 12 
###         0.867 | 0.000 | 0.000 | 0.000 | 13 
###         0.867 | 0.000 | 0.000 | 0.000 | 14 
###         0.867 | 0.000 | 0.000 | 0.000 | 15 
###         0.867 | 0.000 | 0.000 | 0.000 | 16 
###         0.867 | 0.000 | 0.000 | 0.000 | 17 
###         0.867 | 0.000 | 0.000 | 0.000 | 18 
###         0.867 | 0.000 | 0.000 | 0.000 | 19 

### LINEAR SUPPORT VECTOR MACHINE
# from sklearn.svm import LinearSVC
# classifier = LinearSVC()
### ++++++++++++++++++ RESULTS +++++++++++++++++
### Scoring: Recall
###     Features selected: 8
###     Score: 0.364
###     Execution time (1000 iterations): 207.4s
###         Accur | Preci | Recll | F1_sc | features
###         0.788 | 0.117 | 0.228 | 0.133 | 02 
###         0.802 | 0.107 | 0.197 | 0.120 | 03 
###         0.796 | 0.210 | 0.294 | 0.213 | 04 
###         0.789 | 0.221 | 0.334 | 0.233 | 05 
###         0.763 | 0.214 | 0.350 | 0.230 | 06 
###         0.726 | 0.181 | 0.356 | 0.211 | 07 
###         0.711 | 0.184 | 0.364 | 0.215 | 08 <---
###         0.710 | 0.169 | 0.329 | 0.198 | 09 
###         0.695 | 0.145 | 0.316 | 0.178 | 10 
###         0.704 | 0.133 | 0.292 | 0.164 | 11 
###         0.717 | 0.156 | 0.297 | 0.181 | 12 
###         0.737 | 0.174 | 0.321 | 0.200 | 13 
###         0.737 | 0.200 | 0.343 | 0.224 | 14 
###         0.741 | 0.216 | 0.344 | 0.234 | 15 
###         0.738 | 0.219 | 0.345 | 0.235 | 16 
###         0.737 | 0.210 | 0.340 | 0.226 | 17 
###         0.732 | 0.215 | 0.343 | 0.229 | 18 
###         0.737 | 0.208 | 0.338 | 0.226 | 19

### RANDOM FOREST
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier()
### ++++++++++++++++++ RESULTS +++++++++++++++++
### Scoring: Recall
###     Features selected: 3
###     Score: 0.225
###     Execution time (1000 iterations): 274.3s
###         Accur | Preci | Recll | F1_sc | features
###         0.831 | 0.187 | 0.157 | 0.162 | 02 
###         0.866 | 0.326 | 0.225 | 0.255 | 03 <---
###         0.859 | 0.306 | 0.213 | 0.241 | 04 
###         0.859 | 0.276 | 0.182 | 0.212 | 05 
###         0.858 | 0.233 | 0.158 | 0.180 | 06 
###         0.856 | 0.218 | 0.147 | 0.169 | 07 
###         0.858 | 0.221 | 0.139 | 0.164 | 08 
###         0.854 | 0.225 | 0.154 | 0.176 | 09 
###         0.855 | 0.222 | 0.147 | 0.171 | 10 
###         0.854 | 0.197 | 0.136 | 0.154 | 11 
###         0.854 | 0.200 | 0.129 | 0.152 | 12 
###         0.855 | 0.209 | 0.136 | 0.158 | 13 
###         0.855 | 0.179 | 0.117 | 0.137 | 14 
###         0.858 | 0.192 | 0.122 | 0.145 | 15 
###         0.854 | 0.197 | 0.128 | 0.150 | 16 
###         0.857 | 0.203 | 0.130 | 0.153 | 17 
###         0.857 | 0.210 | 0.133 | 0.157 | 18 
###         0.855 | 0.186 | 0.122 | 0.142 | 19 

### ADABOOST
# from sklearn.ensemble import AdaBoostClassifier
# classifier = AdaBoostClassifier()
### ++++++++++++++++++ RESULTS +++++++++++++++++
### Scoring: Recall
###     Features selected: 3
###     Score: 0.275
###     Execution time (1000 iterations): 1137.4s
###         Accur | Preci | Recll | F1_sc | features
###         0.831 | 0.252 | 0.225 | 0.224 | 02 
###         0.826 | 0.285 | 0.275 | 0.263 | 03 <---
###         0.829 | 0.235 | 0.212 | 0.211 | 04 
###         0.834 | 0.249 | 0.217 | 0.220 | 05 
###         0.834 | 0.260 | 0.215 | 0.224 | 06 
###         0.833 | 0.251 | 0.207 | 0.215 | 07 
###         0.834 | 0.248 | 0.206 | 0.213 | 08 
###         0.833 | 0.257 | 0.208 | 0.218 | 09 
###         0.832 | 0.256 | 0.214 | 0.221 | 10 
###         0.838 | 0.278 | 0.235 | 0.241 | 11 
###         0.839 | 0.290 | 0.237 | 0.247 | 12 
###         0.844 | 0.324 | 0.264 | 0.275 | 13 
###         0.849 | 0.330 | 0.267 | 0.279 | 14 
###         0.848 | 0.323 | 0.265 | 0.275 | 15 
###         0.846 | 0.316 | 0.262 | 0.271 | 16 
###         0.846 | 0.319 | 0.268 | 0.275 | 17 
###         0.844 | 0.317 | 0.271 | 0.275 | 18 
###         0.844 | 0.314 | 0.267 | 0.272 | 19 

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
IMPORTANT NOTES: 
    
RESULT:
    
********************************************************************************
"""

dump_classifier_and_data(clf, my_dataset, features_list)


"""
********************************************************************************
ANNEX 0 - Code used while testing and tunning
IMPORTANT NOTES: 
    
RESULT:
    
********************************************************************************
"""

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

