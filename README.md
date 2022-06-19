<p align="center">
    <img alt="veni" src="imglogo.png" width="300" />
</p>

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/veni/badge/?version=latest)](https://veni.readthedocs.io/en/latest/?badge=latest)

# veni

A Python package for deep learning using forward automatic differentiation based on JAX.

## Table of contents
* [Description](#description)
* [Dependencies and installation](#dependencies-and-installation)
	* [Installing from source](#installing-from-source)
<!-- * [Documentation](#documentation) -->
<!-- * [Testing](#testing) -->
* [Examples and Tutorials](#examples-and-tutorials)
* * [Benchmarks](#benchmarks)
* [References](#references)
	<!-- * [Recent works with PyDMD](#recent-works-with-pydmd) -->
* [Authors and contributors](#authors-and-contributors)
* [How to contribute](#how-to-contribute)
	* [Submitting a patch](#submitting-a-patch)
* [License](#license)

## Description
**veni** is a Python package, built on JAX, providing an easy interface to deal with Neural Network using forward automatic differention. Inspired by the very recent (2021) papers of [Atılım Günes Baydin et al.](https://doi.org/10.48550/arXiv.2202.08587) and [David Silver et al.](https://openreview.net/forum?id=5i7lJLuhTm), we have decided to implement a package able to reproduce the results, and give freedom to further investigate this new emerging area of AI.

## Dependencies and installation
**veni** requires requires `jax`, `jaxlib`, `torch`, `numpy`, `sphinx` (for the documentation). The code is tested for Python 3, while compatibility of Python 2 is not guaranteed anymore. It can be installed directly from the source code.

### Installing from source
The official distribution is on GitHub, and you can clone the repository using
```bash
> git clone https://github.com/DSSC-projects/veni
```
You can also install it using pip via
```bash
> python -m pip install git+https://github.com/DSSC-projects/veni
```
<!-- ## Documentation -->
<!-- **PyDMD** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation. You can view the documentation online [here](http://mathlab.github.io/PyDMD/). To build the html version of the docs locally simply: -->

<!-- ```bash -->
<!-- > cd docs -->
<!-- > make html -->
<!-- ``` -->

<!-- The generated html can be found in `docs/build/html`. Open up the `index.html` you find there to browse. -->


<!-- ## Testing -->

<!-- We are using Travis CI for continuous intergration testing. You can check out the current status [here](https://travis-ci.org/mathLab/PyDMD). -->

<!-- To run tests locally (`pytest` is required): -->

<!-- ```bash -->
<!-- > pytest -->
<!-- ``` -->

## Examples and Tutorials
The directory `examples` contains some examples showing how to use **VENI**. In particular we show how to create simple deep learning architectures, how to train via forward automatic differentiation an architecture, and finally how to sample differently candidate directions.

## Benchmarks
The directory `benchmarks` contains some important benchmarks showing how to reproduce [Atılım Günes Baydin et al.](https://doi.org/10.48550/arXiv.2202.08587) results by using the simple **VENI** interface. We further provide logs for efficient analysis of the data. Further benchmark involving directions and optimizers are also available for testing.

### References
To implement the package we follow these works:

* A. G. Baydin, B. A. Pearlmutter, D. Syme, F. Wood, and P. Torr. _Gradients without back-
propagation, 2022
* D. Silver, A. Goyal, I. Danihelka, M. Hessel, and H. van Hasselt. _Learning by directional
gradient descent. In International Conference on Learning Representations, 2022
* Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018). JAX: composable transformations of Python+NumPy programs (0.3.13) [Computer software]. http://github.com/google/jax


## Authors and contributors
**veni** is currently developed and mantained by [Data Science and Scientific Computing](https://dssc.units.it/) master students:
* [Francesco Tomba](mailto:francesco.tomba17@gmail.com)
* [Dario Coscia](https://github.com/dario-coscia)
* [Alessandro Pierro](https://github.com/AlessandroPierro)


Contact us by email for further information or questions about **VENI**, or suggest pull requests. Contributions improving either the code or the documentation are welcome!


## How to contribute
We'd love to accept your patches and contributions to this project. There are just a few small guidelines you need to follow.

### Submitting a patch

  1. It's generally best to start by opening a new issue describing the bug or
     feature you're intending to fix.  Even if you think it's relatively minor,
     it's helpful to know what people are working on.  Mention in the initial
     issue that you are planning to work on that bug or feature so that it can
     be assigned to you.

  2. Follow the normal process of [forking][] the project, and setup a new
     branch to work in.  It's important that each group of changes be done in
     separate branches in order to ensure that a pull request only includes the
     commits related to that bug or feature.

  3. To ensure properly formatted code, please make sure to use 4
     spaces to indent the code. The easy way is to run on your bash the provided
     script: ./code_formatter.sh. You should also run [pylint][] over your code.
     It's not strictly necessary that your code be completely "lint-free",
     but this will help you find common style issues.
     
  4. Do your best to have [well-formed commit messages][] for each change.
     This provides consistency throughout the project, and ensures that commit
     messages are able to be formatted properly by various git tools.

  5. Finally, push the commits to your fork and submit a [pull request][]. Please,
     remember to rebase properly in order to maintain a clean, linear git history.

[forking]: https://help.github.com/articles/fork-a-repo
[pylint]: https://www.pylint.org/
[coveralls]: https://coveralls.io
[well-formed commit messages]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[pull request]: https://help.github.com/articles/creating-a-pull-request

## Citations

## License

See the [LICENSE](LICENSE.rst) file for license rights and limitations (MIT).
