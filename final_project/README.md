


# Summary

The focus of this project is to try to construct a prediction model that could predict if a person is a POI (person of interest) in the [Enron Scandal](https://en.wikipedia.org/wiki/Enron_scandal). The *starter pack* contains a dataset from Enron with a 19 key features, grouped by:
  - 14 Finantial features
  - 5 Email features
  - 1 POI (boolean feature describing if a given person is or not a POI based on if that person was charge, convicted or link (settlement or testification) to fraud)

This dataset is composed by 146 datapoints (persons), from which 18 are identified as POI. There are two datapoint that must be removed, as they are not persons discovered by performing a visual exploration of the data plus some plots. These datapoints are labeled as "TOTAL" and "THE TRAVEL AGENCY IN THE PARK".

> The final dataset used to perform the investigation is composed by 144 datapoints (18 POI's) and 19 features for each datapoint

The procedure followed, was structured in 3 main stages:

  - Feature Selection
  - Classifier testing & tunning
  - Validate & evaluate

# Feature Selection
To perform this stage, I used the script in the *feature_selection.py* file, plus some helper functions from *helper.py* file, that can be found in the same repository.

The objective is to find best features using GridSearchCV, by testing manually combinations of Classifiers (GaussianNB, DecisionTreeClassifier, SVM, LinearSVC, RandomForestClassifier & AdaBoostClassifier), and Selectors (SelectPercentile, SelectKBest). 

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

> Given these numbers, I decided to select 3 features (exercised_stock_options, total_stock_value, bonus) using the DecisionTreeClassifier, because this classifier, offers the best balance among all the evaluation metrics, plus a good time processing the calculations.

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

And the importances of the 3 selected features using DecisionTreeClassifier was:

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

## 

