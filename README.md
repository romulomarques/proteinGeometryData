# FBS: a Frenquency-Based Search on Protein DDGP binary trees.

Python notebook that generates the dataframe of the experiments in

A probabilistic search on the solution space of the Molecular Distance Geometry Problem,

by Rômulo S. Marques, Michael Souza, Fernando Baptista, Miguel Gonçalves, and Carlile Lavor.

It collects the frequency of

short DDGP binary substrings associated with Nuclear Magnetic Resonance (NMR)

instances available on the [Protein Data Bank (PDB)](https://www.rcsb.org/), 

and calculates the FBS and DFS (Depth Fisrt Search) costs.


To reproduce de experiments, the researcher must run the following scripts in the presented order:

0. We are assuming that the researcher already have the folder 'segments';

1. run create_xbsol.py;

2. run create_prune_edges.py;

3. run create_dmdgp.py;

4. run create_dmdgp_HA9H.py;

5. run solvers/sbbu/run_dfs.py;

6. run create_dmdgp_HA9H_bsol.py;

7. run create_test_train.py;

8. in the sbbu_t constructor (script 'sbbu.h'), change the argument of the function 'read_fbs' to 
the absolute path of the file 'df_train.py' (it was generated after running the step 7.).

9. run solvers/sbbu/run_fbs.py;

Now, to generate the graphics, the researcher has to:

10. run create_speedups.py;

11. run all the cells from the python notebook analysis_dmdgp_HA9H.ipynb.

