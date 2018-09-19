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

#for each fold:
for fold in range(nfolds):
  train_sel=np.array([x!=fold for x in seqs])
  test_sel=np.array([x==fold for x in seqs])
  train=ratings[train_sel]
  test=ratings[test_sel]
  
  #calculate model parameters: mean rating over the training set:
  gmr=np.mean(train[:,2])

  #Create array with average ratings for each user, if the user_ID is not in the training set then give the value 0.
  user_sorted_ratings = train[train[:,0].argsort()]
  user_bincount = np.bincount(user_sorted_ratings[:,0])
  user_cumsum = np.cumsum(user_bincount)
  user_average_ratings = []

  for i in range(len(user_bincount)):
    if user_bincount[i] == 0:
      user_average_ratings.append(0)
    else:
      user_current_ratings = user_sorted_ratings[user_cumsum[i-1]:user_cumsum[i],2]
      user_average_ratings.append(np.mean(user_current_ratings))

  user_average_ratings = np.array(user_average_ratings)

  #Predict the average rating for the user if you have ratings from the user, otherwise give the global average rating. For the training set obviously you will always have one.
  train_prediction_ratings = []
  for i in range(len(train)):
    train_prediction_ratings.append(user_average_ratings[train[i,0]])

  train_prediction_ratings = np.array(train_prediction_ratings)
    
  #apply the model to the train set:
  RMSE_train[fold]=np.sqrt(np.mean((train[:,2]-train_prediction_ratings)**2))
  MAE_train[fold]=np.mean(np.abs(train[:,2]-train_prediction_ratings))

  #Predict the average rating for the movie if you have one for it, otherwise give the global average rating.
  test_prediction_ratings = []

  for i in range(len(test)):
    if test[i,1]<=len(user_average_ratings)-1: #This line is here in case the user_ID is higher than the highest user_ID that had ratings in the training set
      if user_average_ratings[test[i,0]]!=0:
        test_prediction_ratings.append(user_average_ratings[test[i,0]])
      else:
        test_prediction_ratings.append(gmr)
    else:
      test_prediction_ratings.append(gmr)

  test_prediction_ratings = np.array(test_prediction_ratings)
    
  #apply the model to the test set:
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
