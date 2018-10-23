import scipy.sparse as sparse
import numpy as np
import math
import itertools
import sys
import time

def make_bin_matrix(path):
  um_data = np.load(path)

  row = np.array(um_data[:,1]) # Array of the movie_id's of each rating
  col = np.array(um_data[:,0]) # Array of the user_id's of each rating
  data = np.ones(len(um_data)) # The rated movie's will be saved as a one

  # Store data in as a sparse matrix with only 0 and 1's so save a datatype binary to save memory.
  binary_matrix = sparse.csr_matrix((data, (row,col)), shape=(row.max()+1,col.max()+1), dtype='b')
  
  return binary_matrix

def make_sign_matrix(bin_matrix, num_signs):
  sign_matrix = np.zeros((num_signs,bin_matrix.shape[1]))

  # Make signature matrix by permuting rows and finding the indices of the first non zero value for each column.
  for sign in range(num_signs):
    row_permutation = np.random.permutation(bin_matrix.shape[0])
    permuted_matrix = bin_matrix[row_permutation, :]
    sign_matrix[sign] = np.array(permuted_matrix.argmax(axis=0))

  return sign_matrix

def LSH(sign_matrix, b, r, sign_sim):
  num_sign = len(sign_matrix[:,0])
  req_buckets = int(math.floor(num_sign/r))

  # Divide the signature matrix in bands and find similar users by grouping the bands that have the same values for tuples in the same buckets.
  # After that eliminate consider only the buckets that have more than one user and pair them up.
  # If the pairs have the required Jacard similarity in their Signatures they are returned as candidates.
  candidate_pairs = []
  for band in range(req_buckets):
    buckets = {}
    band_indices = range(band*r,(1+band)*r)
    
    for user_id in range(len(sign_matrix[0])):
      h = tuple(sign_matrix[band_indices,user_id])
      if h in buckets:
        buckets[h].append(user_id)
      else:
        buckets[h]=[user_id]
    
    for bucket, user_ids in buckets.items():
      if len(user_ids)>1:
        for pair in itertools.combinations(user_ids,2):
          if controle_sign_matrix(pair) > sign_sim:
            candidate_pairs.append(pair)
            
  return candidate_pairs

def controle_sign_matrix(pair):
  
  # Calculate the Jaccard similarity for a pair for their signature matrix.
  same = float(np.count_nonzero(sign_matrix[:, pair[0]] == sign_matrix[:, pair[1]]))
  return same/len(sign_matrix[:, pair[0]])

def controle_bin_matrix(pairs, req_sim, matrix):

  # Calculate the (real) Jaccard similarity for the list of pairs for their binary matrix.
  # Return a list of pairs that match the similarity criteria.
  matrix = matrix.toarray()
  finalpairs = []
  for i in range(len(pairs)):
    intersec = np.sum(matrix[:, pairs[i][0]] & matrix[:, pairs[i][1]])
    union = np.sum(matrix[:, pairs[i][0]] | matrix[:, pairs[i][1]])
    jac_sim = float(intersec) / float(union)
    
    if jac_sim > req_sim:
      finalpairs.append([pairs[i][0],pairs[i][1]])

  return finalpairs

if __name__ == '__main__':
  start_time = time.time()
  
  seed = int(sys.argv[1])
  path = sys.argv[2]
  
  np.random.seed(seed)

  print('Reading in data and making binary matrix ...')
  OG_binary_matrix = make_bin_matrix(path)

  print('Making signature matrix ...')
  sign_matrix = make_sign_matrix(OG_binary_matrix, 100)

  print('Finding candidate pairs and checking the similarities with the signature matrix ...')
  candidate_pairs = LSH(sign_matrix, 20, 5, 0.5)
  uniq_candidate_pairs = np.unique(candidate_pairs, axis=0)

  print('Checking the final list of candidate pairs with the Jaccard Similarity of the binary matrix ...')
  final_pairs = controle_bin_matrix(uniq_candidate_pairs, 0.5, OG_binary_matrix)

  print('Writing the results to a .txt file ...')
  file = open('results.txt', 'a')
  for pair in final_pairs:
    file.write(str(pair[0]) + "," + str(pair[1]))
    file.write('\n')
  file.close()

  print('Found ' + str(len(final_pairs)) + ' pairs.')
  print('Total run time: ' + str(time.time()-start_time) +' seconds.')
