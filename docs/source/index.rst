.. _documentation:

Welcome to veni's documentation!
=================================
`veni <https://github.com/DSSC-projects/veni>`_ is a Python package, built on JAX, providing an easy interface to deal with Neural Network using forward automatic differention. Inspired by the very recent (2021) papers of Atılım Günes Baydin et al. and David Silver et al., we have decided to implement a package able to reproduce the results, and give freedom to further investigate this new emerging area of AI..

**Applications**

veni is born with the intent to provide an easy interface for building deep learning architectures and train them using forward or backward automatic differentiation. The package is really easy to use, please check the `example page <https://github.com/DSSC-projects/veni/tree/main/examples>`_ to see how to use the main functionalities of veni.

**References**

The interested reader in forward automatic differentiation may found an in-depth discussion in the following paper::

    A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind, “Automatic
    differentiation in machine learning: a survey,” 2015


Bibliography
^^^^^^^^^^^^

The following is a non exaustive list of references which we consulted during the development of the project::

   - A. G. Baydin, B. A. Pearlmutter, D. Syme, F. Wood, and P. Torr. _Gradients without back- propagation, 2022
   
   - D. Silver, A. Goyal, I. Danihelka, M. Hessel, and H. van Hasselt. _Learning by directional gradient descent. In International Conference on Learning
   Representations, 2022
   
   - Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., 
   Zhang, Q. (2018). JAX: composable transformations of Python+NumPy programs (0.3.13) [Computer software]. http://github.com/google/jax
