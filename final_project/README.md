


# Summary

The focus of this project is to try to construct a prediction model that could predict if a person is a POI (person of interest) in the [Enron Scandal](https://en.wikipedia.org/wiki/Enron_scandal). The *starter pack* contains a dataset from Enron with a 19 key features, grouped by:
  - 14 Finantial features
  - 5 Email features
  - 1 POI (boolean feature describing if a given person is or not a POI based on if that person was charge, convicted or link (settlement or testification) to fraud)

This dataset is composed by 146 datapoints (persons), from which 18 are identified as POI. There are two datapoint that must be removed, as they are not persons discovered by performing a visual exploration of the data plus some plots. These datapoints are labeled as "TOTAL" and "THE TRAVEL AGENCY IN THE PARK".

> The final dataset used to perform the investigation is composed by 144 datapoints (18 POI's) and 19 features for each datapoint

The procedure followed, was structured in 3 main stages:

  - Feature Selection & Classifier Testing
  - Classifier tunning & Validation
  - Evaluation

# Feature Selection & Classifier Testing
To perform this stage, I used the script in the *feature_selection.py* file, plus some helper functions from *helper.py* file, that can be found in the same repository.

The objective is to find best features using GridSearchCV, by testing manually combinations of Classifiers (GaussianNB, DecisionTreeClassifier, SVM, LinearSVC, RandomForestClassifier & AdaBoostClassifier), and Selectors (SelectPercentile, SelectKBest).

The Evaluation Metrics used to compare the performace among all the classifiers were:

  - Accuracy: the number of correct classification (poi or not poi in our case) divided by the number of data points. 
  - Precision: the ability of the classifier not to label as positive a sample that is negative.
  - Recall: the ability of the classifier to find all the positive samples.
  - F1: weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

Additionally, the *Time Spent* performing the calculations was stored.

## Scorer
A custom scorer was developed, which stores accuracy, precision, recall and F1 from each number of features, and returns recall as scorer to use inside the evaluation method in GridSearchCV. Changing the return metric to precision, accuracy or F1, GridSearchCV will use this as scorer. 

## Selector
The final selector used was SelectKBest, because allowed me to test an specific number of features, unlike SelectPercentile where I had to choose a % of features on each iteration.

## Classifier
The best scores obtained with the multiple tests performed with several classifiers were:

| Classifier | Accuracy | Precision | Recall | F1 | #Features | Time Spent |
|------------|----------|-----------|--------|----|------------|------------|
|Decision Tree | 0.811 | 0.299 | 0.354 | 0.305 | 03 | 110.5s |
|Naive Bayes | 0.335 | 0.147 | 0.834 | 0.250 | 19 | 109.1s |
|SVM | 0.857 | 0.000 | 0.000 | 0.000 | 02 | 147.4s |
|LinearSVC | 0.711 | 0.184 | 0.364 | 0.215 | 08 | 207.4s |
|Random Forest | 0.866 | 0.326 | 0.225 | 0.255 | 03 | 274.3s |
|AdaBoost | 0.826 | 0.285 | 0.275 | 0.263 | 03 | 1137.4s |

We can observe some interesting things:

  - The time spent by adaboost is near 8 times greater than the other classifiers in average
  - The most coincidence number of features selected is 3
  - The best accuracy achieved was 0.866 by RandomForestClassifier
  - The best precision achieved was 0.326 by RandomForestClassifier
  - The best recall achieved was 0.364 by LinearSVC
  - The best F1 Score achieved ewas 0.305 by DecisionTreeClassifier

> Given these numbers, I decided to select 3 features (exercised_stock_options, total_stock_value, bonus) using the DecisionTreeClassifier, because this classifier, offers the best balance among all the evaluation metrics, plus a good time processing the calculations. Special Mention to RandomForestClassifier, which results are good enought to be considered as final Classifier, but the time spent to perform the calculations is more than 2 times greater than DecisionTreeClassifier.

The final table of weights for each feature was:

| Weight | Feature |
|--------|---------|
| 24.82 | exercised_stock_options |
| 24.18 | total_stock_value |
| 20.79 | bonus |
| 18.29 | salary |
| 9.92 | long_term_incentive |
| 9.21 | restricted_stock |
| 8.77 | total_payments |
| 8.59 | shared_receipt_with_poi |
| 7.18 | loan_advances |
| 6.09 | expenses |
| 5.24 | from_poi_to_this_person |
| 4.19 | other |
| 2.38 | from_this_person_to_poi |
| 2.13 | director_fees |
| 1.65 | to_messages |
| 0.22 | deferral_payments |
| 0.17 | from_messages |
| 0.07 | restricted_stock_deferred |

And the importances of the 3 selected features using DecisionTreeClassifier were:

| Imp | Feature |
|-----|---------|
| 0.41 | total_stock_value |
| 0.31 | exercised_stock_options |
| 0.28 | bonus |

# New Features

The creation of new features, was the part of the project that took me more time to complete, because I tried a lot of features combinations but only few of them improved the model, although the improvement was smaller than I expected.

Finally, two new features were created:

  - *Salary_ratio_log*: Viewing the high weight of the feature *salary*, I decided to test some combinations using it to see if I could improve the model performance. Ploting *salary* versus *total_payments* gave me an idea on how to deal with it, and the logaritmic transformation of the ratio salary / total_payments seems to do a good job.
  - *from_to_poi_ratio*: This is a variable that combines all the email features in one, because I regard that if a person is a poi, could have a higher ratio of messages sent to + received from a poi than a not poi persons. The final feature is the sum of messages sent to, received from and shared receipt with a poi divided by the sum of total amount of messages sent and received.

# Classifier tunning & Validation

As said previously, the final Classifier Used was DecisionTreeClassifier, basically due to the balance among all the evaluation Metrics, but specially the F1 Score. The time spent performing the calculations was an important point on the final decision too.

The in initial values of the evaluation metrics without tunning the classifier were:

| Accuracy | Precision | Recall |   F1  |
|----------|-----------|--------|-------|
|  0.806   |   0.368   | 0.365  | 0.367 |

Tunning a classifier means to find the best parameters to maximize the score.Using a pipeline with all the parameters to be tuned, the classifier, a scorer and a CrossValidation method (StratifiedShuffleSplit), we can launch a GridSearchCV that find best score value for a given scorer. The parameters used to test were:

  - splitter:  ['best','random'],
  - criterion: ['entropy', 'gini'],
  - min_samples_split: [2, 4, 6, 8],
  - max_depth: [2, 4, 6],
  - class_weight: [None, "auto"],
  - max_leaf_nodes: [None] + range(2, 10, 1)

Because of the small size of the dataset, the Cross-Validator StratifiedShuffleSplit was used like the test_classifier function on the *tester.py* script. This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class. The number of folds used were 1000.

Surprisingly, once performed the classifier tunning (after a loooong processing time...), the only parameter that improves the model (different of the default ones) was *splitter* set to *random*, and the results were:

| Accuracy | Precision | Recall |   F1  |
|----------|-----------|--------|-------|
|  0.819   |   0.370   | 0.383  | 0.377 |

> As we can see, tunning the classifier with a different parameters form the default ones gives slightly better results than not tunned version.




