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

def estimate_cardinality(values, k):
  """Estimates the number of unique elements in the input set values.

  Arguments:
    values: An iterator of hashable elements to estimate the cardinality of.
    k: The number of bits of hash to use as a bucket number; there will be 2**k buckets.
  """
  num_buckets = 2 ** k
  max_zeroes = [0] * num_buckets
  for value in values:
    h = hash(value)
    bucket = h & (num_buckets - 1) # Mask out the k least significant bits as bucket ID
    bucket_hash = h >> k
    max_zeroes[bucket] = max(max_zeroes[bucket], trailing_zeroes(bucket_hash))
  return 2 ** (float(sum(max_zeroes)) / num_buckets) * num_buckets * 0.79402 #alpha_m(num_buckets)

print([100000/estimate_cardinality([random.random() for i in range(100000)], 10) for j in range(10)])

"""
for i in range(10):
  example_number = random.getrandbits(32)
  print("{:32b}".format(example_number))
  print(trailing_zeroes(example_number))
  print("")
"""
