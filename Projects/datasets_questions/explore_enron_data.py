#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    featureFormat() and targetFeatureSplit() in tools/feature_format.py
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
#print(enron_data)
#print(len(enron_data))
#print([v for v in enron_data.keys()])
#print([len(v) for v in enron_data.values()])
#print([v for v in enron_data.values()[0]])
#print(sum(v['poi'] for v in enron_data.values()))
#print(enron_data['PRENTICE JAMES']['total_stock_value'])
#print(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])
#print(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])

# for i in ["SKILLING JEFFREY K", "LAY KENNETH L", "FASTOW ANDREW S"]:
#     print(i, enron_data[i]['total_payments'])

# known_salaries = 0
# known_emails = 0
# salaries = [v['salary'] for v in enron_data.values()]
# emails = [v['email_address'] for v in enron_data.values()]
# for salary in salaries:
#     if salary != 'NaN':
#         known_salaries += 1
# for email in emails:
#     if email != 'NaN':
#         known_emails += 1
# print(known_salaries, known_emails)

# not_known_total_payments = 0
# total_payments = [v['total_payments'] for v in enron_data.values()]
# for payment in total_payments:
#     if payment == 'NaN':
#         not_known_total_payments += 1
# print(not_known_total_payments, round(float(not_known_total_payments)/float(len(enron_data)), 3)*100)

# not_known_poi_total_payments = 0
# total_payments = [v['total_payments'] for v in enron_data.values() if v['poi']]
# for payment in total_payments:
#     if payment == 'NaN':
#         not_known_poi_total_payments += 1
# print(not_known_poi_total_payments, round(float(not_known_poi_total_payments)/float(len(enron_data)), 3)*100)

# not_known_total_payments = 0
# total_payments = [v['total_payments'] for v in enron_data.values()]
# for payment in total_payments:
#     if payment == 'NaN':
#         not_known_total_payments += 1
# print(len(enron_data) + 10, not_known_total_payments + 10, round(float(not_known_total_payments + 10)/float(len(enron_data) + 10), 3)*100)

# not_known_poi_total_payments = 0
# num_of_poi = sum(v['poi'] for v in enron_data.values())
# total_payments = [v['total_payments'] for v in enron_data.values() if v['poi']]
# for payment in total_payments:
#     if payment == 'NaN':
#         not_known_poi_total_payments += 1
# print(num_of_poi + 10, not_known_poi_total_payments + 10, round(float(not_known_poi_total_payments + 10)/float(num_of_poi + 10), 3)*100)

