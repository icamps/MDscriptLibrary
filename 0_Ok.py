#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

def radgyr(atomgroup, masses, total_mass=None):
    # coordinates change for each frame
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    # get squared distance from center
    ri_sq = (coordinates-center_of_mass)**2
    # sum the unweighted positions
    sq = np.sum(ri_sq, axis=1)
    sq_x = np.sum(ri_sq[:,[1,2]], axis=1) # sum over y and z
    sq_y = np.sum(ri_sq[:,[0,2]], axis=1) # sum over x and z
    sq_z = np.sum(ri_sq[:,[0,1]], axis=1) # sum over x and y

    # make into array
    sq_rs = np.array([sq, sq_x, sq_y, sq_z])

    # weight positions
    rog_sq = np.sum(masses*sq_rs, axis=1)/total_mass
    # square root and return
    return np.sqrt(rog_sq)

class RadiusOfGyration2(AnalysisBase):  # subclass AnalysisBase

    def __init__(self, atomgroup, verbose=True):
        """
        Set up the initial analysis parameters.
        """
        # must first run AnalysisBase.__init__ and pass the trajectory
        trajectory = atomgroup.universe.trajectory
        super(RadiusOfGyration2, self).__init__(trajectory,
                                               verbose=verbose)
        # set atomgroup as a property for access in other methods
        self.atomgroup = atomgroup
        # we can calculate masses now because they do not depend
        # on the trajectory frame.
        self.masses = self.atomgroup.masses
        self.total_mass = np.sum(self.masses)

    def _prepare(self):
        """
        Create array of zeroes as a placeholder for results.
        This is run before we begin looping over the trajectory.
        """
        # This must go here, instead of __init__, because
        # it depends on the number of frames specified in run().
        self.results = np.zeros((self.n_frames, 6))
        # We put in 6 columns: 1 for the frame index,
        # 1 for the time, 4 for the radii of gyration

    def _single_frame(self):
        """
        This function is called for every frame that we choose
        in run().
        """
        # call our earlier function
        rogs = radgyr(self.atomgroup, self.masses,
                      total_mass=self.total_mass)
        # save it into self.results
        self.results[self._frame_index, 2:] = rogs
        # the current timestep of the trajectory is self._ts
        self.results[self._frame_index, 0] = self._ts.frame
        # the actual trajectory is at self._trajectory
        self.results[self._frame_index, 1] = self._trajectory.time

    def _conclude(self):
        """
        Finish up by calculating an average and transforming our
        results into a DataFrame.
        """
        # by now self.result is fully populated
        self.average = np.mean(self.results[:, 2:], axis=0)
        columns = ['Frame', 'Time (ps)', 'Radius of Gyration',
                   'Radius of Gyration (x-axis)',
                   'Radius of Gyration (y-axis)',
                   'Radius of Gyration (z-axis)',]
        self.df = pd.DataFrame(self.results, columns=columns)

get_ipython().run_line_magic('matplotlib', 'inline')
print(mda.__version__)


# In[ ]:


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


# In[ ]:


#defining SETS
print('Defining sets...')
sel_prot = "segid PROT"
sel_lig = "segid LIG"
sel_water = "segid SOLV"


complexx_temp = u2.select_atoms(sel_prot, sel_lig)
# New Universe just with protein+ligand (dry)
u = mda.Merge(complexx_temp).load_new(
         AnalysisFromFunction(lambda ag: ag.positions.copy(),
                              complexx_temp).run().results,
         format=MemoryReader)
prot = u.select_atoms(sel_prot)
lig = u.select_atoms(sel_lig)
water = u.select_atoms(sel_water)
sel_ca = 'protein and name CA'
sel_bb = 'backbone'
ca = u.select_atoms('protein and name CA') #(C_alphas)
bb = u.select_atoms('backbone') #(Backbone)
complexx = u.select_atoms(sel_prot, sel_lig)
#print (u.atoms)
#print (u.trajectory)
#print (prot)
#print (lig)
#print (water)


# In[ ]:


# Aligning the trajectory to a reference
#print('Aling and wrap')
#average = align.AverageStructure(u, u, select='protein and name CA',ref_frame=0).run(verbose=True)
#ref = average.universe
#aligner = align.AlignTraj(u, ref, select='protein and name CA', in_memory=False).run(verbose=True)

# Wrap

#ag = u.atoms
#transform = mda.transformations.wrap(ag)
#u.trajectory.add_transformations(transform)
#print(lig.atoms)
#print(u.segments)


# In[ ]:


# Write PDB Trajectories
print("Writing Trajectories to PDB")
with MDAnalysis.Writer("out_" + fname_root + "_PDBtrajectories.pdb", bonds=None, multiframe=True) as W:
    for ts in u.trajectory:
        W.write(complexx)


# In[ ]:


# HBond calculation
print('HBond Calculation')
# https://docs.mdanalysis.org/2.0.0-dev0/documentation_pages/analysis_modules.html#hydrogen-bonding

# In case need HBond for specific residue: 
# sel_res = "resid 142"
# hb.to_csv("out_" + fname_root + "_HBonds_" + sel_res + "_Analysis" + ".csv")

hbonds = HBA(universe=u, donors_sel=None, hydrogens_sel=None, acceptors_sel=None, 
              between=[sel_prot,sel_lig])
hbonds.run(verbose=True)

column_names = ["Frame", "Donor_id", "Hydrogen_id", "Acceptor_id", "Distance_Angs", "Angle_degree"]
hb = pd.DataFrame(hbonds.hbonds, columns = column_names)
print(hb)
tau_timeseries, timeseries = hbonds.lifetime()

hb.to_csv("out_" + fname_root + "_HBonds_Analysis" + ".csv")
#with open("out_" + fname_root + "_HBonds_Analysis.out", "w") as txt_file:
#    for line in hbonds.hbonds:
#        txt_file.write("".join(str(line)) + "\n") # works with any number of elements in a line


# In[ ]:


# Histograms
# Histogram of distances
distances = hbonds.hbonds[:, 4]
d_histo, d_edges = np.histogram(distances, density=True)
d_midpoints = 0.5*(d_edges[1:] + d_edges[:-1])

# Plotting Histogram of distance 
plt.figure()
plt.plot(d_midpoints, d_histo, label='Distances ($\AA$)')
plt.legend()
plt.savefig("out_" + fname_root + "_HBonds_HistoDistances.png", bbox_inches='tight', dpi=300)

# Histogram of angles
angles = hbonds.hbonds[:, 5]
a_histo, angles_edges = np.histogram(angles, density=True)
a_midpoints = 0.5*(angles_edges[1:] + angles_edges[:-1])

# Plotting Histogram of angles
plt.figure()
plt.plot(a_midpoints, a_histo, label='Angles (degree)')
plt.legend()
plt.savefig("out_" + fname_root + "_HBonds_HistoAngle.png", bbox_inches='tight', dpi=300)


# In[ ]:


# Ok
# HBond analysis: time, types, residues
counts = hbonds.count_by_time() # HBond vs Time
tcounts = hbonds.count_by_type() # HBond vs Type
idcounts = hbonds.count_by_ids() # HBonds vs IDs
#print(idcounts)
#print(len(idcounts))
#print(tcounts)

# Number of Hbonds vs time
df_tcounts = pd.DataFrame(tcounts,columns=["Donor","Acceptor","Total_Number_Times"])
df_tcounts.to_csv("out_" + fname_root + "_HBonds_types" + ".csv")

# Plotting Hbonds vs time
plt.figure()
plt.plot(hbonds.times, counts, label='HBonds vs time')
plt.savefig("out_" + fname_root + "_HBonds_time.png", bbox_inches='tight', dpi=300)        
        
df_time = pd.DataFrame(hbonds.times,columns=["Time"])
df_count = pd.DataFrame(hbonds.count_by_time(),columns=["Count"])
hbonds_time = pd.DataFrame(columns = ["Time", "Count"])
hbonds_time["Time"] = df_time["Time"]
hbonds_time["Count"] = df_count["Count"]
hbonds_time.to_csv("out_" + fname_root + "_HBonds_time" + ".csv")


# HBonds frequency
pd.options.mode.chained_assignment = None  # default='warn'
column_names = ["Resname", "Resid", "Residue", "Number_of_times", "N_total", "Freq"]
data_hbonds = pd.DataFrame(columns = column_names)

for i in range(len(idcounts)):
    data_hbonds.at[i,'Number_of_times'] = idcounts[i,3]
    idd = str(u.atoms[idcounts[i,0]])
    #print(i, idd)
    m = idd.find(sel_lig)
    #print(m)
    if m == -1: # The first atom is NOT from LIGAND
        rn = re.search('resname (.+?),', idd)
        ResName = rn.group(1)
        data_hbonds.at[i,'Resname'] = ResName
        ri = m = re.search('resid (.+?) ', idd)
        ResNumber = ri.group(1)
        data_hbonds.at[i,'Resid'] = int(ResNumber)
        data_hbonds.at[i,'Residue'] = ResName + ResNumber
    else:
        idd = str(u.atoms[idcounts[i,2]])
        #print(i, idd)
        #print(m)
        rn = re.search('resname (.+?),', idd)
        ResName = rn.group(1)
        data_hbonds.at[i,'Resname'] = ResName
        ri = m = re.search('resid (.+?) ', idd)
        ResNumber = ri.group(1)
        data_hbonds.at[i,'Resid'] = int(ResNumber)
        data_hbonds.at[i,'Residue'] = ResName + ResNumber

data_hbonds.sort_values(by=['Resid'], inplace=True, ascending=True)
data_hbonds['N_total'] = data_hbonds.groupby(['Resname', 'Resid', 'Residue'])['Number_of_times'].transform('sum')
sorted_data_hbonds = data_hbonds.drop_duplicates(subset=['Resname', 'Resid', 'Residue'])
sorted_data_hbonds['Freq'] = sorted_data_hbonds['N_total'] / sorted_data_hbonds['N_total'].sum()

data_hbonds.to_csv("out_" + fname_root + "_HBonds_unsorted" + ".csv")
sorted_data_hbonds.to_csv("out_" + fname_root + "_HBonds_residues" + ".csv")

# Plotting Hbonds frequency
#plt.figure()
ax3 = sorted_data_hbonds.plot(x='Residue', y='Freq', kind='bar', stacked=True, colormap='Paired')
ax3.set_ylabel('Frequency ($\%$)')
ax3.plot()
ax3.figure.savefig("out_" + fname_root + "_HBonds_residues.png", bbox_inches='tight', dpi=300)


# In[ ]:


# RMSD calculation
print('RMSD Calculation: C-alphas')
# C_alphas
R_ca = rms.RMSD(ca, ca, select='protein and name CA', ref_frame=0)
R_ca.run(verbose=True)

# Plotting & Saving files RMSD calculation: 
plt.figure()
df_ca_rmsd = pd.DataFrame(R_ca.rmsd, columns=['Frame', 'Time (ns)', 'ca'])
df_ca_rmsd["Time (ns)"] = 0.001 * df_ca_rmsd["Time (ns)"]
ax = df_ca_rmsd.plot(x='Time (ns)', y='ca', kind='line')
ax.set_ylabel('RMSD ($\AA$)')
df_ca_rmsd.to_csv("out_" + fname_root + "_RMSD_CA" + ".csv")
ax.plot()
plt.savefig("out_" + fname_root + "_RMSD_CA.png", bbox_inches='tight', dpi=300) #2D-RMSD.pdf

# Backbone
print('RMSD Calculation: Backbone')
R_bb = rms.RMSD(bb, bb, select='backbone', ref_frame=0)
R_bb.run(verbose=True)

# Plotting & Saving files RMSD calculation: 
plt.figure()
df_bb_rmsd = pd.DataFrame(R_bb.rmsd, columns=['Frame', 'Time (ns)', 'bb'])
df_bb_rmsd["Time (ns)"] = 0.001 * df_bb_rmsd["Time (ns)"]
ax = df_bb_rmsd.plot(x='Time (ns)', y='bb', kind='line')
ax.set_ylabel('RMSD ($\AA$)')
df_bb_rmsd.to_csv("out_" + fname_root + "_RMSD_BB" + ".csv")
ax.plot()
plt.savefig("out_" + fname_root + "_RMSD_BB.png", bbox_inches='tight', dpi=300) #2D-RMSD.pdf

# Ligand
print('RMSD Calculation: Ligand')
R_LIG = rms.RMSD(lig, lig, select=sel_lig, ref_frame=0)
R_LIG.run(verbose=True)

# Plotting & Saving files RMSD calculation: 
plt.figure()
df_LIG_rmsd = pd.DataFrame(R_LIG.rmsd, columns=['Frame', 'Time (ns)', 'LIG'])
df_LIG_rmsd["Time (ns)"] = 0.001 * df_LIG_rmsd["Time (ns)"]
ax = df_LIG_rmsd.plot(x='Time (ns)', y='LIG', kind='line')
ax.set_ylabel('RMSD ($\AA$)')
df_LIG_rmsd.to_csv("out_" + fname_root + "_RMSD_LIG" + ".csv")
ax.plot()
plt.savefig("out_" + fname_root + "_RMSD_LIG.png", bbox_inches='tight', dpi=300) #2D-RMSD.pdf

#Pairwise RMSD of a trajectory to itself
print('2D-RMSD Calculation')
#aligner = align.AlignTraj(u, u, select='name CA', in_memory=False).run()
matrix = diffusionmap.DistanceMatrix(u, select='name CA').run(verbose=True)
#matrix.dist_matrix.shape

rmsd2d = pd.DataFrame(data = matrix.dist_matrix)
rmsd2d.to_csv("out_" + fname_root + "_2D-RMSD" + ".csv")

# Plot Pairwise RMSD of a trajectory to itself
plt.figure()
plt.imshow(matrix.dist_matrix, cmap='viridis')
plt.xlabel('Frame')
plt.ylabel('Frame')
plt.colorbar(label=r'RMSD ($\AA$)')
plt.savefig("out_" + fname_root + "_2D-RMSD_CA.png", bbox_inches='tight', dpi=300) #2D-RMSD.pdf


# RMSF Calculation
print('RMSF Calculation')
R_ca_rmsf = rms.RMSF(ca).run(verbose=True)

# Plotting RMSF
plt.figure()
plt.plot(ca.resids, R_ca_rmsf.rmsf)
plt.xlabel('Residue number')
plt.ylabel('RMSF ($\AA$)')

df_ca_rmsf = pd.DataFrame(R_ca_rmsf.rmsf, columns=['ca.resids'])
plt.savefig("out_" + fname_root + "_RMSF.png", bbox_inches='tight', dpi=300)
df_ca_rmsf.to_csv("out_" + fname_root + "_RMSF" + ".csv")

# Save PDB with RMSF as temperture factor
column = df_ca_rmsf["ca.resids"]
max_tempfactor = column.max()
min_tempfactor = column.min()
column = df_ca_rmsf["ca.resids"]
max_tempfactor = column.max()
min_tempfactor = column.min()
data_tempfactor = [['Minimum', min_tempfactor], ['Maximum', max_tempfactor]] 
df = pd.DataFrame(data_tempfactor, columns = ['Label', 'tempFactor']) 
df.to_csv("out_" + fname_root + "_RMSF_tempfactors.txt")

u.add_TopologyAttr('tempfactors') # add empty attribute for all atoms
for residue, r_value in zip(prot.residues, R_ca_rmsf.rmsf):
    residue.atoms.tempfactors = r_value

u.atoms.write("out_" + fname_root + "_RMSF_tempfactors.pdb") # Warning are ok

# Visualize
#view = nv.show_mdanalysis(u)
#view.update_representation(color_scheme='bfactor')
#view


# In[ ]:


# Radius of gyration
print('RadGyr Calculation')
rog_base = RadiusOfGyration2(prot, verbose=True).run(verbose=True)

# Plotting & Saving files:
plt.figure()
ax = rog_base.df.plot(x='Time (ps)', y=rog_base.df.columns[2:])
ax.set_ylabel('Radius of gyration (A)');
ax.plot()
plt.savefig("out_" + fname_root + "_RadGyr.png", bbox_inches='tight', dpi=300)
rog_base.df.to_csv("out_" + fname_root + "_RadGyr" + ".csv")


# In[ ]:


#Pairwise RMSD of a trajectory to itself
# Backbone
print('2D-RMSD Calculation backbone')
#aligner = align.AlignTraj(u, u, select=sel_bb, in_memory=False).run()
matrix_bb = diffusionmap.DistanceMatrix(u, select=sel_bb).run(verbose=True)
#matrix.dist_matrix.shape

rmsd2d_bb = pd.DataFrame(data = matrix_bb.dist_matrix)
rmsd2d_bb.to_csv("out_" + fname_root + "_2D-RMSD_BB" + ".csv")

# Plot Pairwise RMSD of a trajectory to itself
plt.figure()
plt.imshow(matrix_bb.dist_matrix, cmap='viridis')
plt.xlabel('Frame')
plt.ylabel('Frame')
plt.colorbar(label=r'RMSD ($\AA$)')
plt.savefig("out_" + fname_root + "_2D-RMSD_BB.png", bbox_inches='tight', dpi=300)

#C-alpha
print('2D-RMSD Calculation C-alpha')
#aligner = align.AlignTraj(u, u, select=sel_ca, in_memory=False).run()
matrix_ca = diffusionmap.DistanceMatrix(u, select=sel_ca).run(verbose=True)
#matrix.dist_matrix.shape

rmsd2d_ca = pd.DataFrame(data = matrix_ca.dist_matrix)
rmsd2d_ca.to_csv("out_" + fname_root + "_2D-RMSD_CA" + ".csv")

# Plot Pairwise RMSD of a trajectory to itself
plt.figure()
plt.imshow(matrix_ca.dist_matrix, cmap='viridis')
plt.xlabel('Frame')
plt.ylabel('Frame')
plt.colorbar(label=r'RMSD ($\AA$)')
plt.savefig("out_" + fname_root + "_2D-RMSD_CA.png", bbox_inches='tight', dpi=300)

