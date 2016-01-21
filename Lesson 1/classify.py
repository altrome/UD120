def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()
    
    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)
    coin = 0
    for i in range(len(labels_test)):
        if (labels_test[i] == pred[i]):
            coin += 1
        #print('Index: ' + str(i) + ' Prediction: ' + str(pred[i]) + ' Original value: ' + str(labels_test[i]))



    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = round((float(coin) / float(len(labels_test))), 2)
    return accuracy

    #from sklearn.metrics import accuracy_score
    #print accuracy_score(pred,labels_test)