""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    xMin = min(arr)
    xMax = max(arr)
    i = 0
    for value in arr:
        arr[i] = round(float(value - xMin) / float(xMax - xMin), 3)
        i += 1
    return arr

# tests of your feature scaler--line below is input data
# data = [115, 140, 175]
# print featureScaling(data)

# Min/Max Scaler in sklearn
from sklearn.preprocessing import MinMaxScaler
import numpy
weights = numpy.array([[115.0], [140.0], [175.0]])
scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights)
print rescaled_weight