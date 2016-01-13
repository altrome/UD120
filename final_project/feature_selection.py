#!/usr/bin/python

'''
********************************************************************************
FEATURE SELECTION
DESCRIPTION:
    The objective is to find best features using GridSearchCV:
    Tests done manually using combinations of:
    Classifiers 
        GaussianNB
        DecisionTreeClassifier
        SVM
        LinearSVC 
        RandomForestClassifier
        AdaBoostClassifier
    Selectors
        SelectPercentile 
        SelectKBest
    Scorers
        Custom Selector getScoring()
USE:
    Uncomment the two lines of code next to selector name you want to use, 
        - default DecisionTreeClassifier
    Uncomment the two lines of code next to Classifier name you weant to use,
        - default SelectKbest
    Go to helpers and choose the scorer that GridSearchCV will use to evaluate
    the classifier, by changing the return statement (line 156)
        - default recall
RESULT:
    A table with different metrics obtained from GridSearchCV
OPTIONAL:
    Using DecisionTreeClassifier, Uncomment last part of code to print a weight 
    table of all features
********************************************************************************
'''

import sys
import pickle
from math import isnan
sys.path.append("../tools/")
import warnings
warnings.filterwarnings("ignore")

from feature_format import featureFormat, targetFeatureSplit

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### List of all possible features + poi first
features_list = [   'poi',
                    'salary', 
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
                    'shared_receipt_with_poi',
                    # 'salary_ratio',
                    # 'deferred_ratio',
                    # 'extras',
                    'salary_ratio_log',
                    'from_to_poi_ratio',
                    # 'restricted_stock_real'
                    ]

### Removing outliers identified in poi_id TASK 0
for outlier in ['TOTAL','THE TRAVEL AGENCY IN THE PARK']:
    data_dict.pop(outlier,0)

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

my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

### SELECTOR: SelectPercentile (varing percentile from 10 to 100 with step 5)
# from sklearn.feature_selection import SelectPercentile, f_classif
# selector = SelectPercentile(f_classif)

### SELECTOR: SelectKbest (varing k from 2 to 19 with step 1)
from sklearn.feature_selection import SelectKBest
selector = SelectKBest()

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
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
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


from helper import getScorings, printMetrics

pipeline = Pipeline([('selector', selector),('classifier', classifier)])
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
parameters = {  
                # 'selector__percentile': range(10, 101, 5) # for Percentile
                'selector__k': range(2, 20, 1) # for Kbest
             }
clf = GridSearchCV( pipeline, 
                    parameters, 
                    cv = cv,
                    scoring = getScorings # precision, recall, f1, getScorings
                    ) 

clf.fit(features, labels)

printMetrics(clf)

print '\nBest score is', clf.best_score_, 'with', clf.best_estimator_.get_params()['selector__k'], 'features'

### Uncomment for DecisionTreeClassifier
print "\nfeatures weight\n"
scores = clf.best_estimator_.named_steps['selector'].scores_
print "Weight\tFeature"
print "------\t-------"
ordered_features = []
for score in sorted(enumerate(scores), key=lambda x:x[1], reverse=True):
    ordered_features.append(features_list[score[0]+1])
    print " %.2f \t%s" % (score[1], features_list[score[0]+1])
print '\nclassifier features importance\n'
importances = clf.best_estimator_.named_steps['classifier'].feature_importances_
print "  imp \tFeature"
print "------\t-------"
for imp in sorted(enumerate(importances), key=lambda x:x[1], reverse=True):
    print " %.2f \t%s" % (imp[1], ordered_features[imp[0]])

# for fold in clf.grid_scores_:
#     print fold



# Weight  Feature
# ------  -------
#  24.82  exercised_stock_options
#  24.18  total_stock_value
#  20.79  bonus
#  18.29  salary
#  9.92   long_term_incentive
#  9.21   restricted_stock
#  8.77   total_payments
#  8.59   shared_receipt_with_poi
#  7.18   loan_advances
#  6.09   expenses
#  5.24   from_poi_to_this_person
#  4.19   other
#  2.38   from_this_person_to_poi
#  2.13   director_fees
#  1.65   to_messages
#  0.22   deferral_payments
#  0.17   from_messages
#  0.07   restricted_stock_deferred

