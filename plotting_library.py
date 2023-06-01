#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from scipy import optimize
import pandas as pd
import scienceplots


# In[6]:


# Set Plotting Style
plt.style.use(['science','nature','vibrant'])

# Modify The Plotting Style
# Modify the default plot settings
plt.rcParams['figure.figsize'] = (6,6/1.333)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = False
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.25
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
# Colors
seq_map = ['#fc9272','#fb6a4a','#ef3b2c','#cb181d','#99000d']
seq_map = ['#000000','#377eb8','#4daf4a','#e41a1c']

            # Gold    # Purple  #Grey     # Black  
pur_map = ['#CFB991','#977886','#555960','#000000']
psu_map = ['#E98424','#1C3F7A','#BD2A56','#999999','#000000']
dblue = np.array([30, 144, 255])/255
fgreen = np.array([75, 122, 71])/255
purp = np.array([138,43,226])/255
yell = np.array([253, 218, 13])/255
black = np.array([0,0,0])

# Get the Set1 color map
cmap = plt.get_cmap('Set1')


# In[ ]:




