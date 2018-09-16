
"""
  Created on Fri Sep 18 21:09:10 2015
  
  @author: Wojtek Kowalczyk
  
  This script demonstrates how implement the "global average rating" recommender
  and validate its accuracy with help of 5-fold cross-validation.
  
  """

import numpy as np

#load data
#ratings=read_data("ratings.dat")
ratings=[]
f = open("datasets/ratings.dat", 'r')
for line in f:
  data = line.split('::')
  ratings.append([int(z) for z in data[:3]])
f.close()
ratings=np.array(ratings)

"""
  Alternatively, instead of reading data file line by line you could use the Numpy
  genfromtxt() function. For example:
  
  ratings = np.genfromtxt("ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int')
  
  will create an array with 3 columns.
  
  Additionally, you may now save the rating matrix into a binary file
  and later reload it very quickly: study the np.save and np.load functions.

"""


#split data into 5 train and test folds
nfolds=5

#allocate memory for results:
err_train=np.zeros(nfolds)
err_test=np.zeros(nfolds)

#to make sure you are able to repeat results, set the random seed to something:
np.random.seed(17)

seqs=[x%nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)

num_factors = 10
num_iter = 75
labda = 0.05
eta = 0.005

#for each fold:
for fold in range(nfolds):
  train_sel=np.array([x!=fold for x in seqs])
  test_sel=np.array([x==fold for x in seqs])
  train=ratings[train_sel]
  test=ratings[test_sel]

  #Define the User and Movie features matrices with num_factors factors
  User_features_matrix = np.random.rand(np.max(ratings[:,0]),num_factors)
  Movie_features_matrix = np.random.rand(num_factors,np.max(ratings[:,1]))

  #Iterate num_iter times over all available ratings in the training set and update the User and Movie features
  #according to the equations in gravity-Tikk
  for iteration in range(num_iter):
    for i in range(len(train)):
      error = train[i,2] - np.dot(User_features_matrix[train[i,0]-1],Movie_features_matrix[:,train[i,1]-1])
      User_features_matrix[train[i,0]-1] += eta*(2*error*Movie_features_matrix[:,train[i,1]-1]-labda*User_features_matrix[train[i,0]-1])
      Movie_features_matrix[:,train[i,1]-1] += eta*(2*error*User_features_matrix[train[i,0]-1]-labda*Movie_features_matrix[:,train[i,1]-1])

  #Use the features to make predictions on the training set and cut off ratings lower than 1 and higher than 5.
  train_prediction_ratings = []

  for i in range(len(train)):
    prediction = np.dot(User_features_matrix[train[i,0]-1],Movie_features_matrix[:,train[i,1]-1])
    if prediction < 1:
      train_prediction_ratings.append(1)
    elif prediction > 5:
      train_prediction_ratings.append(5)
    else:
      train_prediction_ratings.append(prediction)

  train_prediction_ratings = np.array(train_prediction_ratings)

  err_train[fold]=np.sqrt(np.mean((train[:,2]-train_prediction_ratings)**2))

  #Use the features to make predictions on the test set and cut off ratings lower than 1 and higher than 5.
  test_prediction_ratings = []

  for i in range(len(test)):
    prediction = np.dot(User_features_matrix[train[i,0]-1],Movie_features_matrix[:,train[i,1]-1])
    if prediction < 1:
      test_prediction_ratings.append(1)
    elif prediction > 5:
      test_prediction_ratings.append(5)
    else:
      test_prediction_ratings.append(prediction)

  test_prediction_ratings = np.array(test_prediction_ratings)

  err_test[fold]=np.sqrt(np.mean((test[:,2]-test_prediction_ratings)**2))
 
  #print errors:
  print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))

#print the final conclusion:
print("\n")
print("Mean error on TRAIN: " + str(np.mean(err_train)))
print("Mean error on  TEST: " + str(np.mean(err_test)))

# Just in case you need linear regression: help(np.linalg.lstsq) will tell you
# how to do it!
