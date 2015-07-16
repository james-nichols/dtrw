DTRW
====

Simulations of continuous time random walks (CTRWs) and factional partial differential equations through the discrete time random walk (DTRW) method. The theory and methods are outlined in the paper "A discrete time random walk model for anomalous diffusion" by C.N. Angstmann, I.C. Donnelly, B.I. Henry, J.A. Nichols, in the Journal of Computaional Physics. See http://www.sciencedirect.com/science/article/pii/S002199911400549X

This repository contains code that replicates results in several papers, currently in preparation. The publication list will be updated shortly.

TODO:
-----
- Clean up and streamline compartment model code
- Potentially do multiple inheritance for subdiffusive vs diffusive classes, or have diffusive and non-diffusive in the one class...
- Include knowledge of Delta T and Delta X in the DTRW class, so spatial parameters are abstracted within the class, perhaps even within some factory class...
