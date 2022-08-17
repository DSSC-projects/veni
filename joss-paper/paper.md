---
title: 'veni: A Python package for deep learning using forward automatic differentiation in JAX'
tags:
  - Python
  - deep learning
  - automatic differentiation
authors:
  - name: Dario Coscia
    equal-contrib: true
    affiliation: "1, 2" 
  - name: Alessandro Pierro
    equal-contrib: true 
    affiliation: 1
  - name: Francesco Tomba
    equal-contrib: true 
    affiliation: 1
affiliations:
  - name: Internation School of Advanced Studies, SISSA, Trieste, Italy
    index: 1
  - name: Università degli Studi di Trieste
    index: 2
date: 17 August 2022
bibliography: paper.bib

---


# Summary

Automatic Differentiation (AD) is a general framework for algorithmically computing the exact derivative (or gradient) of a function specified as a computer program [@ad_survey], involving two main methods: backward-mode automatic differentiation (BWD), and forward-mode automatic differentiation (FWD). Parameter optimization in deep learning (DL) relies heavily on BWD, which is well enstablished and implemented in many DL libraries. Recent researches [@gradient_without_backprop;@silver2021learning] have developed a new framework involving FWD for parameter optimization, which speeds up the training and testing process in a DL pipeline, while mantaining the same performance as in BWD. Nevertheless, current DL libaries do not support FWD optimization, making it hard to reproduce and extend the cited works.

`veni` is a Python package for deep learning using forward automatic differentiation in JAX. The package provides an easy interface to deal with Neural Networks optimization using FWD. Furthermore, `veni` works with BWD as well, allowing a fair comparison for gradient calculation methods. By the virtue of being written in JAX, `veni` natively supports DL models to run on CPU and GPUs, as well as efficient vectorised operations. Consequently, `veni` can be used by researchers and scientists to investigate new approaches to parameter optimization by using AD and their main methods.

# Statement of need

Current deep learning optimization methods for network parameters optimization rely on back-propagation, also known as backward-mode automatic differentiation (BWD). Being the state of the art, major deep learning libraries as PyTorch or TensorFlow are build on the dependency of BWD for optimizing network parameters. Nevertheless, BWD is part of a broader area of differentiation strategies called automatic differentiation (AD). Automatic differentiation is mainly divided in BWD and forward automatic differentiation (FWD), which differ on the way the chain rule is computed. For the interest reader we suggest [@ad_survey] for a complete overview on the topic. Recent papers developed independently by Deep Mind researchers [@silver2021learning] and Oxford researchers [@gradient_without_backprop], have applied the FWD technique in a deep learing optimization setting, obtaing faster results than BWD in training and testing deep learning models, while maintaining same level of performances. Nevertheless, since most deep learning libriaries only support BWD, reproducing the results or expaning and testing new strategies based on FWD is rather challening. 

`veni` is a Python package for deep learning, based on JAX, which provides an easy interface for network optimization based on FWD, BWD or both. By being based on JAX, `veni` can support efficient training and testing on CPU and GPUs as well as efficient strategies for performing AD. In `veni` the main computational deep learning layers, loss functions and optimizers are implemented. The API for `veni` is designed to provide a class-based and userfriendly interface to build deep learning networks, and optimize network parameters through automatic differentiation. Since neural network optimization based on forward-mode AD is still an under-explored topic in deep learning, the `veni` package is intended for researchers and scientits who want to explore this new area of study.


# Acknowledgements 

We acknowledge the support recieved by Ansuini Alessio from the Università degli Studi di Trieste during the preliminar theoretical study of the topic, Nicola Demo from the SISSA mathLab group for advices on the paper creation, as well as the ORFEO data center for providing the hardware to perform experiments on GPUs for large networks.


# References
