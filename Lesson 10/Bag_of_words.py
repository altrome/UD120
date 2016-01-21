from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
string1 = "Hi Katie the self driving car will be late Best Sebastian"
string2 = "Hi Sebastian the machine learning class will be great great great Best Katie"
string3 = "Hi Katie the machine learning class will be most excellent"
email_list = [string1, string2, string3]
bag_of_words = vectorizer.fit(email_list)
bag_of_words = vectorizer.transform(email_list)
#return word frequency
print bag_of_words
#return word id
print vectorizer.vocabulary_.get("great")

