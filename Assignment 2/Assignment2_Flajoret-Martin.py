import numpy as np
import timeit
import random
import math

def trailing_zeroes(num):
  """Counts the number of trailing 0 bits (least significant) in num."""
  if num == 0:
    return 32 # Assumes 32 bit integer inputs!
  p = 0
  """>>shifts binary to the right by deleting most right end.
  & is bitwise AND operator only gives a 1 for 1&1 otherwise 0"""
  while (num >> p) & 1 == 0:
    p += 1

return p

def alpha_m(m):
  a = -1+2**(-1/m)
  b = math.gamma(-1/m)
  return (a*b/math.log(2))**-m

def average_group_estimate_cardinality(values):
  """Estimates the number of unique elements in the input set values.

  Arguments:
    values: An iterator of hashable elements to estimate the cardinality of.
    k: The number of bits of hash to use as a bucket number; there will be 2**k buckets.
  """
  values = np.array(values)
  
  num_groups = int(len(values)/np.log2(len(values)))
  max_zeroes = np.zeros(num_groups)

  seqs=[x%num_groups for x in range(len(values))]
  np.random.shuffle(seqs)

  for group in range(num_groups):
    group_values = values[np.array([x==group for x in seqs])]
    for value in group_values:
      h = hash(value)
      max_zeroes[group] = max(max_zeroes[group], trailing_zeroes(h))

  return np.average(2 ** max_zeroes)

values=[random.random() for i in range(100000)]

values = np.array(values)
num_groups = int(len(values)/np.log2(len(values)))
max_zeroes = np.zeros(num_groups)

seqs=[x%num_groups for x in range(len(values))]
np.random.shuffle(seqs)

for group in range(num_groups):
  group_values = values[np.array([x==group for x in seqs])]
  for value in group_values:
    h = hash(value)
    max_zeroes[group] = max(max_zeroes[group], trailing_zeroes(h))
  print(group)

def mean_average_group_estimate_cardinality(values, mean_size):
  averages_array = np.array([average_group_estimate_cardinality(values) for i in range(mean_size)])
  return np.mean(averages_array)

#print([100000/mean_average_group_estimate_cardinality([random.random() for i in range(100000)], 1) for j in range(10)])

"""
for i in range(10):
  example_number = random.getrandbits(32)
  print("{:32b}".format(example_number))
  print(trailing_zeroes(example_number))
  print("")
"""
