import numpy as np
import timeit
import random
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def estimate_cardinality(values):
  """Estimates the number of unique elements in the input set values.

  Arguments:
    values: An iterator of hashable elements to estimate the cardinality of.
    k: The number of bits of hash to use as a bucket number; there will be 2**k buckets.
  """
  max_zeroes = 0
  for value in values:
    max_zeroes = max(max_zeroes, trailing_zeroes(value))

  return 2 ** max_zeroes

def gen_ran_32_bit_num(number_of_values):
  return [random.getrandbits(32) for i in range(number_of_values)]

def median_of_group_averages_estimate_cardinality(num_values, num_groups):
  groups = [[] for i in range(num_groups)]
  group_averages = []
  
  """
  Each group atleast a small multiple of np.log2(m) estimates.
  For each estimate in the group a new hash function, or in our case a new random set of 32 bit integers of length num_values.
  """
  
  for group in range(num_groups):
    for i in range(int(2*np.log2(num_values))):
      groups[group].append(estimate_cardinality(gen_ran_32_bit_num(num_values)))
    group_averages.append(np.median(groups[group]))

  return np.average(group_averages)

m = [i * 10 for i in range(1,11)]#, 2*10**4, 3*10**4]
n = 10**4
local_estimates = [[] for i in range(len(m))]
variances = []
RAE_results_experiment = []

for i in range(len(m)):
  for l in range(10):
    local_estimates[i].append(median_of_group_averages_estimate_cardinality(n, m[i]))
  print(i)

local_estimates = np.array(local_estimates)

for i in range(len(m)):
  variances.append(np.sum((local_estimates[i]-n)**2)/len(local_estimates[i]))
  RAE_results_experiment.append((1/n)*(variances[i]**0.5))

def RAE_func(x, a):
  return a/np.sqrt(x)

optim_parameter, covar_parameter = curve_fit(RAE_func, n, RAE_results_experiment)

mdata = np.linspace(m[0], m[9], 100)
plt.scatter(m, RAE_results_experiment, s=4, c='r')
plt.plot(mdata)
plt.xlim(m[0],m[9])
plt.ylim(0, np.max(RAE_results_experiment))
plt.xlabel(r'$Number\ of\ groups$')
plt.ylabel(r'$RAE$')
#plt.legend()
plt.show()

"""
fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs.hist(local_estimates[4], bins=20)

xlabel = ['Estimate of cardinality']
ticks = [range(int(np.min(local_estimates[4])),int(np.max(local_estimates[4])),100)]

axs.set(xlabel=xlabel[0], ylabel='Counts', xticks = ticks[0])

plt.show()
"""
