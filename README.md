DTRW
====

Simulations of continuous time random walks (CTRWs) and factional partial differential equations through the discrete time random walk (DTRW) method. The theory and methods are outlined in the paper "A discrete time random walk model for anomalous diffusion" by C.N. Angstmann, I.C. Donnelly, B.I. Henry, J.A. Nichols, in the Journal of Computaional Physics. See http://www.sciencedirect.com/science/article/pii/S002199911400549X

This repository contains code that replicates results in several papers, currently in preparation. The publication list will be updated shortly.

TODO:
-----
- Solve using arrival densities methods, as of recently, no longer gives the same results, see e.g. compare_diffusion_methods.py script.
- Complete the compartment model library for FSIR and other simulations.
- Include knowledge of Delta T and Delta X in the DTRW class, so spatial parameters are abstracted within the class, perhaps even within some factory class...
- Remove the has_spatial_reactions logic, and simplify the inheritance tree of diffusive vs subdiffusive DTRW classes, potentially allowing for multiple inheritance for various features, e.g. for subdiffusion with two-layer transitions....
