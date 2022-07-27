import time
import numpy as np
from numpy import genfromtxt
import csv

def give_inputs(sz):

  my_data = genfromtxt('data.csv')
  
  x1 = my_data[0:4096]
  x2 = my_data[4096:]
  
  in3 = (1,sz,sz)
   
  size = (1,in3[0],in3[1],in3[2])
  
  y = []
  y.append(np.reshape(x1,size))
  y.append(np.reshape(x2,size))  
  return y
  
