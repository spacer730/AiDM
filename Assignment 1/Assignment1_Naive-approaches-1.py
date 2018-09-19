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
    
  #apply the model to the train set:
  RMSE_train[fold]=np.sqrt(np.mean((train[:,2]-gmr)**2))
  MAE_train[fold]=np.mean(np.abs(train[:,2]-gmr))
    
  #apply the model to the test set:
  RMSE_test[fold]=np.sqrt(np.mean((test[:,2]-gmr)**2))
  MAE_test[fold]=np.mean(np.abs(test[:,2]-gmr))

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
