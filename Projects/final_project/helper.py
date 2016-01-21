from prettytable import PrettyTable
import sys
sys.path.append("../tools/")

'''
FUNCTION NAME:
    printMetrics(classifier)
ARGUMENTS:
    classifier: result classifier from GridSearchCV
TYPE:
    Helper
DESCRIPTION:
    This function print the result data into a Pretty Table
'''

def printMetrics(classifier):
    print '\n'
    header = ['Accuracy', 'Precision', 'Recall', 'F1']
    header = fillMetrics(classifier.grid_scores_[0], header, True)
    table = PrettyTable(header)
    global metrics
    for key, metric in metrics.items():
        accuracy = sum(metric['accuracy'])/len(metric['accuracy'])
        precision = sum(metric['precision'])/len(metric['precision'])
        recall = sum(metric['recall'])/len(metric['recall'])
        f1 = sum(metric['f1'])/len(metric['f1'])
        row = [accuracy, precision, recall, f1]
        row = fillMetrics(classifier.grid_scores_[key], row)
        table.add_row(row)
    table.float_format = '4.3'
    table.int_format = '2'
    print table
    del metrics

'''
FUNCTION NAME:
    fillMetrics(metrics, row, header = False)
ARGUMENTS:
    metrics: dict with elements to fill
    row: preformated list
    header: boolean, true if should be header
TYPE:
    Helper
DESCRIPTION:
    This function fill the data into a list and return the list
'''

def fillMetrics(metrics, row, header = False):
    for param, value in metrics.parameters.items():
        if header:
            row.append(param)
        else:
            row.append(value)
    return row

'''
FUNCTION NAME:
    GetScorings(estimator, X_test, y_test)
ARGUMENTS:
    estimator: model that should be evaluated, 
    X_test: validation data
    y_test: is the ground truth target for X_test (in the supervised case)
TYPE:
    Custom Scorer
DESCRIPTION:
    This scorer prints accuracy, precision, recall and F1 from
    each value of k, and returns recall as scorer to use
    inside the evaluation method in GridSearchCV. 
USE:
    Changing the return metric to precision, accuracy or F1, 
    GridSearchCV will use this as scorer
RETURNS:
    Scorer
SOURCE:
'''

def getScorings(estimator, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # calculate predictions
    predictions = estimator.predict(X_test)
    # claculate Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    # Number of features
    try:
        features = estimator.get_params()['selector'].get_params()['k']
    except KeyError:
        features = estimator.get_params()
    # Classifier parameters
    params = estimator.get_params()
    # global variable to store metrics and CV parameters on each iteration
    global metrics
    try:
        metrics
    except NameError:
        metrics = {}
        metrics[0] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'params': params
        }
  
    # global variable to store last iteration k
    global lastk
    try:
        lastk
    except NameError:
        lastk = 0
        print 'Starting Grid Search'
        print '_'*50
        print features,
    # Variable to identify fold number
    mKey = len(metrics)-1
    # if we are in a new iteration in GridSearchCV initialize lastk 
    if (features < lastk and lastk > 0):
        lastk = 0
        print '[DONE]'
        print '_'*50
        print features
        mKey += 1
        
    # if mkey not exists initialize it
    if (mKey not in metrics):
        metrics[mKey] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'params': params
        }

    # append iteration metrics in metrics[key]
    metrics[mKey]['accuracy'].append(accuracy)
    metrics[mKey]['precision'].append(precision)
    metrics[mKey]['recall'].append(recall)
    metrics[mKey]['f1'].append(f1)
    
    # if new k initialize metrics[mkey]
    if (features > lastk):      
        if lastk > 0:
            print features,
            mKey += 1
            metrics[mKey] = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'params': params
            }
        lastk = features
    return recall # accuracy, precision, recall or f1

'''
FUNCTION NAME:
    printPrecisionRecall(clf, data, feature_list, folds=1000)
DESCRIPTION:
    This is a copy of test_Classifier function in the tester, but customized 
    to print a Pretty table
'''

def getMetrics(clf, data, feature_list, folds=1000):
    from feature_format import targetFeatureSplit
    from sklearn.cross_validation import StratifiedShuffleSplit
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        # print clf
        # print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        # print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        # print ""
        print "\n"
        header = ['Accuracy', 'Precision', 'Recall', 'F1', 'F2']
        table = PrettyTable(header)
        table.add_row([accuracy, precision, recall, f1, f2])
        table.float_format = '4.3'
        print table
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

    