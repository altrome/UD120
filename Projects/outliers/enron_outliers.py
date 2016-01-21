#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

# ****** Task 2: remove a key-value TOTAL from a data 
data_dict.pop( "TOTAL", 0 ) 

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below

# ****** Task 0: ploting salary vs bonus
# print
# for point in data:
#     salary = point[0]
#     bonus = point[1]
#     matplotlib.pyplot.scatter( salary, bonus )

# matplotlib.pyplot.xlabel("salary")
# matplotlib.pyplot.ylabel("bonus")
# matplotlib.pyplot.show()

# Result: There's a point clearly outside the group (salary ~ 2.5e7 & bonus ~ 1e8)
# Action: Identify who is...

# ****** Task 1: Identifying the biggest outlier
# import math
# maxSalary = 0
# maxBonus = 0
# maxInsider = ""
# for key in data_dict:
#     if (data_dict[key]["salary"] > maxSalary and not math.isnan(float(data_dict[key]["salary"]) )):
#         maxSalary = data_dict[key]["salary"]
#         maxBonus = data_dict[key]["bonus"]
#         maxInsider = key
# print "The insider ", maxInsider, " has the max Salary with value ", maxSalary, " and a bonus of ", maxBonus

# Result: The insider  TOTAL  has the a Salary of  26704229  and a bonus of  97343619
# Action: I need to remove the TOTAL from the original data

# ****** Task 3: Identifying Two More Outliers
import math
limitSalary = 1000000
limitBonus = 5000000
maxInsider = ""
for key in data_dict:
    if (not math.isnan(float(data_dict[key]["bonus"])) and not math.isnan(float(data_dict[key]["salary"]))):
        if ( data_dict[key]["salary"] > limitSalary and data_dict[key]["bonus"] > limitBonus ):
            print "the insider ", key, " has the a Salary of ", data_dict[key]["salary"], " and a bonus of ", data_dict[key]["bonus"]