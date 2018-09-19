import numpy as np
import timeit

start = timeit.default_timer()

#load data
#ratings=read_data("ratings.dat")
ratings=[]
f = open("datasets/ratings.dat", 'r')
for line in f:
  data = line.split('::')
  ratings.append([int(z) for z in data[:3]])
f.close()
ratings=np.array(ratings)

#split data into 5 train and test folds
nfolds=5

#allocate memory for results:
RMSE_train=np.zeros(nfolds)
RMSE_test=np.zeros(nfolds)
MAE_train=np.zeros(nfolds)
MAE_test=np.zeros(nfolds)

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

  RMSE_train[fold]=np.sqrt(np.mean((train[:,2]-train_prediction_ratings)**2))
  MAE_train[fold]=np.mean(np.abs(train[:,2]-train_prediction_ratings))

  #Use the features to make predictions on the test set and cut off ratings lower than 1 and higher than 5.
  test_prediction_ratings = []

  for i in range(len(test)):
    prediction = np.dot(User_features_matrix[test[i,0]-1],Movie_features_matrix[:,test[i,1]-1])
    if prediction < 1:
      test_prediction_ratings.append(1)
    elif prediction > 5:
      test_prediction_ratings.append(5)
    else:
      test_prediction_ratings.append(prediction)

  test_prediction_ratings = np.array(test_prediction_ratings)

  RMSE_test[fold]=np.sqrt(np.mean((test[:,2]-test_prediction_ratings)**2))
  MAE_test[fold]=np.mean(np.abs(test[:,2]-test_prediction_ratings))
 
  #print errors:
  print("Fold " + str(fold) + ": RMSE_train=" + str(RMSE_train[fold]) + "; RMSE_test=" + str(RMSE_test[fold]))
  print("Fold " + str(fold) + ": MAE_train=" + str(MAE_train[fold]) + "; MAE_test=" + str(MAE_test[fold]))

#print the final conclusion:
print("\n")
print("Mean error of RMSE on TRAIN: " + str(np.mean(RMSE_train)))
print("Mean error of RMSE on TEST: " + str(np.mean(RMSE_test)))
print("\n")
print("Mean error of MAE on TRAIN: " + str(np.mean(MAE_train)))
print("Mean error of MAE on TEST: " + str(np.mean(MAE_test)))

stop = timeit.default_timer()
print("\n")
print('Time: ', stop - start)
