#!/usr/bin/env python
# coding: utf-8

# In[1]:


import MDAnalysis as mda
import MDAnalysis
from MDAnalysis.analysis import contacts
import MDAnalysis.transformations
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysisData import datasets
from MDAnalysis.analysis import diffusionmap, align, rms
from MDAnalysis.coordinates.memory import MemoryReader
import MDAnalysis.analysis.pca as pca
import MDAnalysis.analysis.rdf
from MDAnalysis.analysis import rdf
from MDAnalysis.analysis import diffusionmap, align, rms
from MDAnalysis.analysis.base import (AnalysisBase,
                                      AnalysisFromFunction,
                                      analysis_class)
import pandas as pd
import numpy as np
import nglview as nv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import re
import warnings
# suppress some MDAnalysis warnings about writing PDB files
warnings.filterwarnings('ignore')

print(mda.__version__)


# In[3]:


# Input files
fname_root = "test"
fname_Topology = fname_root + ".psf"
fname_Trajectory = fname_root + ".dcd"
fname_PDB = fname_root + ".pdb"
u2 = mda.Universe(fname_Topology, fname_Trajectory, in_memory=False) #, in_memory=False, in_memory_step=500)
#u.transfer_to_memory(start=3000, stop=3025, verbose=True) 
#from MDAnalysis.tests.datafiles import PSF, DCD
# From test suite u = mda.Universe(PSF, DCD)
print(u2.segments)
#u.segments.segids[0]
#u.segments.segids[1]
#ref = mda.Universe(fname_Topology, fname_PDB)
#frag_protein = "segid PROT"
#frag_lig = "segid LIG"

