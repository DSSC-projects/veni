Welcome to veni's documentation!
=================================

A Python package for deep learning using forward automatic
differentiation based on JAX.

Table of contents
-----------------

-  `Description <#description>`__

-  `Dependencies and installation <#dependencies-and-installation>`__

   -  `Installing from source <#installing-from-source>`__

-  `Documentation <#documentation>`__

-  `Examples and Tutorials <#examples-and-tutorials>`__

-  

   -  `Benchmarks <#benchmarks>`__

-  `References <#references>`__

-  `Authors and contributors <#authors-and-contributors>`__

-  `How to contribute <#how-to-contribute>`__

   -  `Submitting a patch <#submitting-a-patch>`__

-  `License <#license>`__

Description
-----------

**veni** is a Python package, built on JAX, providing an easy interface
to deal with Neural Network using forward automatic differention.
Inspired by the very recent (2021) papers of `Atılım Günes Baydin et
al. <https://doi.org/10.48550/arXiv.2202.08587>`__ and `David Silver et
al. <https://openreview.net/forum?id=5i7lJLuhTm>`__, we have decided to
implement a package able to reproduce the results, and give freedom to
further investigate this new emerging area of AI.

Dependencies and installation
-----------------------------

**veni** requires requires ``jax``, ``jaxlib``, ``torch``, ``numpy``,
``sphinx`` (for the documentation). The code is tested for Python 3,
while compatibility of Python 2 is not guaranteed anymore. It can be
installed directly from the source code.

Installing from source
~~~~~~~~~~~~~~~~~~~~~~

The official distribution is on GitHub, and you can clone the repository
using

.. code:: bash

   > git clone https://github.com/DSSC-projects/veni

You can also install it using pip via

.. code:: bash

   > python -m pip install git+https://github.com/DSSC-projects/veni

Documentation
-------------

**veni** uses `Sphinx <http://www.sphinx-doc.org/en/stable/>`__ for code
documentation. You can view the documentation online
`here <https://veni.readthedocs.io/en/latest/>`__. To build the html
version of the docs locally simply:

.. code:: bash

   cd docs
   make html

The generated html can be found in ``docs/build/html``. Open up the
``index.html`` you find there to browse.

Examples and Tutorials
----------------------

The directory `examples <https://github.com/DSSC-projects/veni/tree/main/examples>`__ contains some examples showing how to use
**veni**. In particular we show how to create simple deep learning
architectures, how to train via forward automatic differentiation an
architecture, and finally how to sample differently candidate
directions.

Benchmarks
----------

The directory `benchmarks <https://github.com/DSSC-projects/veni/tree/main/benchmarks>`__ contains some important benchmarks showing
how to reproduce `Atılım Günes Baydin et
al. <https://doi.org/10.48550/arXiv.2202.08587>`__ results by using the
simple **veni** interface. We further provide logs for efficient
analysis of the data. Further benchmark involving directions and
optimizers are also available for testing.

References
~~~~~~~~~~

To implement the package we follow these works:

-  A. G. Baydin, B. A. Pearlmutter, D. Syme, F. Wood, and P. Torr.
   \_Gradients without back- propagation, 2022
-  D. Silver, A. Goyal, I. Danihelka, M. Hessel, and H. van Hasselt.
   \_Learning by directional gradient descent. In International
   Conference on Learning Representations, 2022
-  Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C.,
   Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J.,
   Wanderman-Milne, S., & Zhang, Q. (2018). JAX: composable
   transformations of Python+NumPy programs (0.3.13) [Computer
   software]. http://github.com/google/jax
-  Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array
   programming with NumPy. Nature 585, 357–362 (2020). DOI:
   10.1038/s41586-020-2649-2.
-  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury,
   Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca
   Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito,
   Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner,
   Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative
   style, high-performance deep learning library. In H. Wallach, H.
   Larochelle, A. Beygelzimer, F. d’Alch ́e-Buc, E. Fox, and R. Garnett,
   editors, Advances in Neural Information Processing Systems 32, pages
   8024–8035. Curran Associates, Inc., 2019.

Authors and contributors
------------------------

**veni** is currently developed and mantained by `Data Science and
Scientific Computing <https://dssc.units.it/>`__ master students:
   - `Francesco Tomba <mailto:francesco.tomba17@gmail.com>`__  
   - `Dario Coscia <https://github.com/dario-coscia>`__ 
   - `Alessandro Pierro <https://github.com/AlessandroPierro>`__

Contact us by email for further information or questions about **veni**,
or suggest pull requests. Contributions improving either the code or the
documentation are welcome!

How to contribute
-----------------

We’d love to accept your patches and contributions to this project.
There are just a few small guidelines you need to follow.

Submitting a patch
~~~~~~~~~~~~~~~~~~

1. It’s generally best to start by opening a new issue describing the
   bug or feature you’re intending to fix. Even if you think it’s
   relatively minor, it’s helpful to know what people are working on.
   Mention in the initial issue that you are planning to work on that
   bug or feature so that it can be assigned to you.

2. Follow the normal process of
   `forking <https://help.github.com/articles/fork-a-repo>`__ the
   project, and setup a new branch to work in. It’s important that each
   group of changes be done in separate branches in order to ensure that
   a pull request only includes the commits related to that bug or
   feature.

3. To ensure properly formatted code, please make sure to use 4 spaces
   to indent the code. The easy way is to run on your bash the provided
   script: ./code_formatter.sh. You should also run
   `pylint <https://www.pylint.org/>`__ over your code. It’s not
   strictly necessary that your code be completely “lint-free”, but this
   will help you find common style issues.

4. Do your best to have `well-formed commit
   messages <http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html>`__
   for each change. This provides consistency throughout the project,
   and ensures that commit messages are able to be formatted properly by
   various git tools.

5. Finally, push the commits to your fork and submit a `pull
   request <https://help.github.com/articles/creating-a-pull-request>`__.
   Please, remember to rebase properly in order to maintain a clean,
   linear git history.

Citations
---------

If you are considering using **veni** on your reaserch please cite us:

APA:

.. code:: bash

   Tomba, F., Coscia, D., & Pierro, A. (2022). veni (Version 0.0.1) [Computer software]. https://github.com/DSSC-projects/veni

BibTex:

.. code:: bash

   @software{Tomba_veni_2022,
   author = {Tomba, Francesco and Coscia, Dario and Pierro, Alessandro},
   month = {6},
   title = {{veni}},
   url = {https://github.com/DSSC-projects/veni},
   version = {0.0.1},
   year = {2022}
   }


License
-------

See the `LICENSE <https://github.com/DSSC-projects/veni/blob/main/LICENSE.rst>`__ file for license rights and
limitations (MIT).

See more...
-----------

.. toctree::
   :maxdepth: 2

   usage
   modules