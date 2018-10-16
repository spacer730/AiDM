import scipy.sparse as sparse
import numpy as np
import timeit
import random
import math
import itertools

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

local_binary_matrix = sparse.coo_matrix((data, (row,col)))#, shape=(row.max()+1,col.max()+1)).toarray()
local_binary_matrix = local_binary_matrix.tocsr()

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
  local_binary_matrix = local_binary_matrix.tocsr()
  sign_matrix = np.zeros((num_signs,col.max()+1))
  
  for sign in range(num_signs):
    row_permutation = np.random.permutation(row.max()+1)
    permuted_matrix = local_binary_matrix[row_permutation, :]
    sign_matrix[sign] = np.array(permuted_matrix.argmax(axis=0))
    print(sign)

  return sign_matrix

def LSH(sign_matrix, b, r):
  num_sign = len(sign_matrix[:,0])
  req_buckets = int(math.floor(num_sign/r))
  candidate_pairs = []
  for band in range(req_buckets):
    buckets = {}
    band_indices = range(band*r,(1+band)*r)
    for user_id in range(len(sign_matrix[0])):
      h = hash(tuple(sign_matrix[band_indices,user_id]))
      if h in buckets:
        buckets[h].append(user_id)
      else:
        buckets[h]=[user_id]
    for bucket, user_ids in buckets.items():
      if len(user_ids)>1:
        for pair in itertools.combinations(user_ids,2):
          candidate_pairs.append(pair)
    print(band)
  for band in range(req_buckets,b):
    buckets = {}
    band_indices = random.sample(range(num_sign), r)
    for user_id in range(len(sign_matrix[0])):
      h = hash(tuple(sign_matrix[band_indices,user_id]))
      if h in buckets:
        buckets[h].append(user_id)
      else:
        buckets[h]=[user_id]
    for bucket, user_ids in buckets.items():
      if len(user_ids)>1:
        for pair in itertools.combinations(user_ids,2):
          candidate_pairs.append(pair)
    print(band)
  return candidate_pairs

def jaccard_similarity(user_1, user_2):
   s1 = set(local_binary_matrix[:,user_1].nonzero()[0])
   s2 = set(local_binary_matrix[:,user_2].nonzero()[0])
   intersection = s1.intersection(s2)
   union = s1.union(s2)
   return float(len(intersection)/len(union))

def controle(pairs):
   finalpairs = []
   for i in range(500):#len(pairs)):
      if jaccard_similarity(int(pairs[i][0]), int(pairs[i][1])) > 0.5:
         finalpairs.append(pairs[i])
      print(i)
   return finalpairs

"""
def controle2(pairs):
  finalpairs = []
  for i in range(len(pairs)):
    if jaccard_similarity_score(local_binary_matrix[:,pairs[i][0]], local_binary_matrix[:,pairs[i][1]]) > 0.5:
      finalpairs.append(pairs[i])
    print(i)
  return finalpairs
"""

sign_matrix = make_sign_matrix(row, col, data, 50)
candidate_pairs = LSH(sign_matrix, 5, 11)
uniq_candidate_pairs = np.unique(candidate_pairs, axis=0)
final_pairs = controle(uniq_candidate_pairs)
