DTRW
====

Simulations subdiffusive random processes through the discrete time random walk (DTRW) method. The theory and methods are outlined in the papers 
 - "A discrete time random walk model for anomalous diffusion", C.N. Angstmann, I.C. Donnelly, B.I. Henry, J.A. Nichols, Journal of Computational Physics 2015, http://www.sciencedirect.com/science/article/pii/S002199911400549X
 - "From stochastic processes to numerical methods: A new scheme for solving reaction subdiffusion fractional partial differential equations", C.N. Angstmann, I.C. Donnelly, B.I. Henry, B.A. Jacobs, T.A.M. Langland, J.A. Nichols, Journal of Computational Physics 2016, http://www.sciencedirect.com/science/article/pii/S0021999115007937
 - "Subdiffusive discrete time random walks via Monte Carlo and subordination", JA Nichols, BI Henry, CN Angstmann, Journal of Computational Physics, Submitted 2017, https://arxiv.org/abs/1711.06197

This repository contains code that replicates results in these papers.

TODO:
-----
- Clean up and streamline compartment model code
- Potentially do multiple inheritance for subdiffusive vs diffusive classes, or have diffusive and non-diffusive in the one class...
- Include knowledge of Delta T and Delta X in the DTRW class, so spatial parameters are abstracted within the class, perhaps even within some factory class... consider factory-based production pattern
