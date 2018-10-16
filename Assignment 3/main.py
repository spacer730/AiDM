import scipy.sparse as sparse
import numpy as np
import timeit
import random
import math

np.random.seed(42)

"""
with load('user_movie.npy') as data:
    a = data['a']
"""

"""
Step 1. Load the data and convert it to the binary matrix format.
"""
um_data = np.load('user_movie.npy')

row = np.array(um_data[:,1]) #Array of the movie_id's of each rating
col = np.array(um_data[:,0]) #Array of the user_id's of each rating
data = np.ones(len(um_data)) #The rated movie's will be saved as a one

#binary_matrix = sparse.coo_matrix((data, (row,col)))#, shape=(row.max()+1,col.max()+1)).toarray()

"""
def make_sign_matrix(binary_matrix[:,0], num_signs):
  sign_matrix=np.zeros((num_signs,len(col)))

  for sign in range(num_signs):
    #row = np.random.permutation(len(rows))
    matrix_rows_permutation = np.random.permutation(len(binary_matrix[:,0]))
    
    for user_id in range(len(binary_matrix[0])):
      sign_matrix[sign,user_id]=np.argmax(binary_matrix[:,user_id]) #Returns first index that is one for the permutation of that user_id column
    print(sign)
  return sign_matrix
"""

def make_sign_matrix(row, col, data, num_signs):
  local_binary_matrix = sparse.coo_matrix((data, (row,col)))
  local_binary_matrix = binary_matrix.tocsr()
  sign_matrix = np.zeros((num_signs,col.max()+1))
  
  for sign in range(num_signs):
    row_permutation = np.random.permutation(row.max()+1)
    permuted_matrix = local_binary_matrix[row_permutation, :]
    sign_matrix[sign] = np.array(permuted_matrix.argmax(axis=0))
    print(sign)

  return sign_matrix

sign_matrix = make_sign_matrix(row, col, data, 50)

def LSH(sign_matrix, b, r, k):
  buckets = np.zeros(k)
  req_buckets = math.floor(sign_matrix[:,0]/b)

  
  
  for band in range(req_buckets):
    buckets = {}
    band_indices = range(band*r,(1+band)*r)
    for user_id in range(len(sign_matrix[0])):
      h = hash(sign_matrix[band_indices,user_id].tolist())
      buckets['s']=buckets['s'].append(223)
      if h in buckets:
        buckets.[h]
    
  
  for band in range(b):
    band_indices = np.random.
    for user_id in range(len(sign_matrix[0])):
      hash.


  return candidate_pairs

candidate_pairs = LSH(sign_matrix, 5, 10,
