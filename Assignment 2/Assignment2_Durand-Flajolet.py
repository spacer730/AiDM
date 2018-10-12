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
  & 1 is bitwise AND operator only gives a 1 if num >> p ends with a 1 otherwise 0"""
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
  values_bucket = [[] for i in range(num_buckets)]
  for value in values:
    h = hash(value)
    bucket = h & (num_buckets - 1) # Mask out the k least significant bits as bucket ID
    bucket_hash = h >> k
    values_bucket[bucket].append(bucket_hash)
    max_zeroes[bucket] = max(max_zeroes[bucket], trailing_zeroes(bucket_hash))
<<<<<<< HEAD
  return 2 ** (float(sum(max_zeroes)) / num_buckets) * num_buckets * alpha_m(num_buckets)
  #return max_zeroes, values_bucket

def RAE(E, n):
  return np.abs(E-n)/n

def gen_ran_32_bit_num(number_of_values):
  return [random.getrandbits(32) for i in range(number_of_values)]

n = [10**4]#, 2*10**4, 3*10**4]
k = [i+1 for i in range(2,10)]
m = [2**(i+1) for i in range(2,10)]

local_estimates = [[[] for i in range(len(m))] for i in range(len(n))]
cardinalities = [[[] for i in range(len(m))] for i in range(len(n))]
variances = [[] for i in range(len(n))]
RAE_results_experiment = [[] for i in range(len(n))]

for i in range(len(n)):
  for j in range(len(m)):
    for l in range(1000):
      values = gen_ran_32_bit_num(n[i])
      cardinalities[i][j].append(len(np.unique(values)))
      local_estimates[i][j].append(estimate_cardinality(values, k[j]))
  print(i)
    
cardinalities = np.array(cardinalities)
local_estimates = np.array(local_estimates)

for i in range(len(n)):
  for j in range(len(m)):
    variances[i].append(np.sum((local_estimates[i][j]-cardinalities[i][j])**2)/len(local_estimates[i][j]))
    RAE_results_experiment[i].append((1/n[i])*((variances[i][j])**0.5))

def RAE_func(x, a):
  return a/np.sqrt(x)

parameters = []
for i in range(len(n)):
  optim_parameter, covar_parameter = curve_fit(RAE_func, m, RAE_results_experiment[i])
  parameters.append(optim_parameter[0])

mdata = np.linspace(0.01, 2**10, 100)
plt.scatter(m, RAE_results_experiment[0], s=3, c='r')
plt.plot(mdata, RAE_func(mdata, parameters[0]), label=r'$\frac{'+str(round(parameters[0],2))+'}{\sqrt{m}}$')
plt.xlim(0,2**10)
plt.ylim(0, np.max(RAE_results_experiment[0]))
plt.xlabel(r'$m \ (number of buckets)$')
plt.ylabel(r'$RAE \ $')
plt.legend()
plt.show()
=======
  return 2 ** (float(sum(max_zeroes)) / num_buckets) * num_buckets * 0.79402 #alpha_m(num_buckets)

print([100000/estimate_cardinality([random.random() for i in range(100000)], 10) for j in range(10)])
>>>>>>> ada92d8f7b1344ab01462a7440ca2fc8fb6f3275

"""
#This coding is for making a plot of the histogram of the estimates.

fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs.hist(local_estimates[0][1], bins=30)

xlabel = ['Estimate of cardinality']
ticks = [range(int(np.min(local_estimates[0][1])),int(np.max(local_estimates[0][1])),1000)]

axs.set(xlabel=xlabel[0], ylabel='Counts', xticks = ticks[0])

plt.show()
"""
