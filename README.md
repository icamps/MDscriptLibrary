# scriptLibrary

These scripts are intended to be used with MDAnalyis package to do several analysis af Molecular Dynamics simulations.

**Note**: you need to have the topology and trajectories files from you preferable MD package.

If you don't know the segment names, you can use the script *get_segment_names* to identify them. With the segment names of the protein, ligand and solvent, you add them to the second script doMDAnalysis.

The script doMDAnalysis will carry RMSD, RMSF, RadGyr, 2D-RMSD and CPA analysis, saving CSV files (to be used with your favorite plotting software and their respective images (in PNG format). These images files are intended to be used just as a guide, not as final/production images.
