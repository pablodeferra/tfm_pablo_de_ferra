#%%
import numpy as np
import healpy as hp
import os
from tqdm import tqdm
import pymaster as nmt
import sys
import matplotlib.pyplot as plt

M = np.ones([2,2])

M = np.array([[1,2,3],
     [1,-7,3],
     [1,2,5]])

print(np.linalg.slogdet(M))

print(np.eye(M.shape[-1]))