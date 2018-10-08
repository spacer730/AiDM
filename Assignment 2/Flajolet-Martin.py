#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 23:10:52 2018

@author: Gideon
"""
import random
import numpy as np

#generate arry with random bitstrings of length 32
def generate_stream(length):
   stream = []
   for x in range(length):
      stream.append(random.getrandbits(32))
   return stream

#calculate number of trailing zeroes in bitstring for given num
def trailing_zeroes(num):
  """Counts the number of trailing 0 bits in num."""
  if num == 0:
    return 32 # Assumes 32 bit integer inputs!
  p = 0
  while (num >> p) & 1 == 0:
    p += 1
  return p

def est_distinct(stream):  
   #calculate trailing zeroes for all input value
   zeroes = []
   R = 0
   for y in range(1,len(stream)):
      zeroes.append(trailing_zeroes(stream[y]))
      if trailing_zeroes(stream[y]) > R:
         R = trailing_zeroes(stream[y])
         
   estimation = 2 ** R
   truecount = len(np.unique(stream))
   RAE = (np.abs(estimation - truecount) / truecount)
   
   print('Estimation =' + str(estimation))
   print('True count =' + str(truecount))
   print('RAE =' + str(RAE))

stream = generate_stream(10000)
stream = np.array(stream)
est_distinct(stream)
