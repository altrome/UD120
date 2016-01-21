## UD120 Class Summary

### Dataset / Question
* Do I have enough data?
* Can I define a question?
* Do I have enough / correct features to answer the question?

### Features
* Exploration
    * Inspect for Correlations
    * Outlier Removal
    * Imputation
    * Cleaning
* Creation
    * Think about it like a human would
* Representation
    * Test Vectorization
    * Discretiation
* Scaling
    * Mean Subtraction
    * MinMax Scalar
    * Standard Scalar
* Selection
    * KBest
    * Percentile
    * Recursive Feature Elimination
* Transforms
    * PCA
    * ICA

### Algorithms
* Tune Your Algorithm
    * Parameters of Algorithm
    * Visual Inspection
    * Performance on Test Data
    * GridSearchCV
* Pick an Algorithm
    * Labeled Data (Supervised Classification)
        * Ordered Data (Continuous Output)
            * Linear Regression
            * Lasso Regression
            * Decision Tree Regression
            * SV Regression
        * Unordered Data (Discrete Output)
            * Decision Tree
            * [Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html)
            ```python
            from sklearn.naive_bayes import GaussianNB
            from sklearn.metrics import accuracy_score

            clf = GaussianNB()  # init GaussianNB class on variable clf

            t0 = time()  # start time to train
            clf.fit(features_train, labels_train)  # fit the classifier
            print("training time: ", round(time()-t0, 3), "s")  # output time to fit

            t1 = time()  # start time to predict
            pred = clf.predict(features_test)  # run the prediction
            print("prediction time: ", round(time()-t1, 3), "s")  # output time to predict

            accuracy = accuracy_score(labels_test, pred)  # calculate the accuracy
            ```
            * [SVM (Support Vector Machine)](http://scikit-learn.org/stable/modules/svm.html)
            ```python
            from sklearn import svm
            from sklearn.metrics import accuracy_score
            clf = svm.SVC(C=10000.0, kernel="rbf", gamma=0.0)  # call svm classifier

            t0 = time()  # start time to train
            clf.fit(features_train, labels_train)  # fit the classifier
            print("training time: ", round(time()-t0, 3), "s")  # output time to fit

            t1 = time()  # start time to predict
            pred = clf.predict(features_test)  # run the prediction
            print("prediction time: ", round(time()-t1, 3), "s")  # output time to predict

            accuracy = accuracy_score(labels_test, pred)  # calculate the accuracy
            ```
            * Ensembles
            * K Nearest Neighbors
            * LDA
            * Logistic Regression
    * Unlabeled Data (Unsupervised Classification)
        * K-Means Clustering
        * Spectral Clustering
        * PCA
        * Mixture Models / EM Algorithm
        * Outlier Detection

### Evaluation
* Validate
    * Train / Test Split
    * K-Fold
    * Visualize
* Pick Metric(s)
    * SSE / r^2
    * Precsiion
    * Recall
    * F1 Score
    * ROC Curve
    * Custom Bias / Variance Metric

### Flowchart
![flowchart](Final_chart.jpeg "End of Class Flowchart")