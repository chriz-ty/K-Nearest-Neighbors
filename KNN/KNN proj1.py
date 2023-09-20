import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbours(data, predict, k=3):
    if len(data) >=k:  #checks if k has value less than the total data length.
        warnings.warn('K is set to the value less than the total voting groups!')
    distances = []  #used to store the distances between the input data point.

#calculating Euclidian Distance between two data points.
    for group in data:
        for features in data[group]:
            eucildean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([eucildean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]  #sorts the distances list and selects the k nearest neighbors.
    vote_result = Counter(votes).most_common(1)[0][0]  #count the occurrences of each class label in the votes list and returns the most common class label.
    confidence = Counter(votes).most_common(1)[0][1] / k  # calculates the confidence level of the prediction.

    return vote_result, confidence

df = pd.read_csv("breast-cancer.data.txt")  #reading the sample dataset.
df.replace('?',-99999, inplace=True)
df.drop(['id'],axis=1,inplace=True)
full_data = df.astype(float).values.tolist()  #df is converted to a NumPy array of float values and then converted to a Python list.

random.shuffle(full_data)  #shuffles the order of the data.
test_size=0.2

#creating test set and train set datas.
train_set = {2: [], 4:[]}
test_set =  {2: [], 4:[]}

#These lines split the full_data into training and testing datasets.
#train_data contains the first 80% of the shuffled data, and test_data contains the last 20%.
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])  # append the feature vector excluding the class label as train_set.
for i in test_data:
    test_set[i[-1]].append(i[:-1])  # append the feature vector excluding the class label as test_set.

correct,total=0,0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbours(train_set, data, k=5)  #applying KNN Algorithm
        if group == vote:   #finding correct results
            correct+=1
        total+=1
print('Accuracy: ', correct/total)
