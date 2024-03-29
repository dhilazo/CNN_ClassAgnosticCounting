# Study of Deep Learning Models for Class-Agnostic Counting
<p align="center">
<a href="https://www.python.org/downloads/release/python-374/"><img alt="Python Version" src="https://img.shields.io/badge/python-3.7.4-green"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="Pytorch Version" src="https://img.shields.io/badge/pytorch-1.8.0-red?color=red&label=pytorch&logo=pytorch"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Master's thesis for the <a href="https://www.fib.upc.edu/en/studies/masters/master-artificial-intelligence">Master in Artificial Intelligence</a> at the Polytechnic University of Catalonia (UPC). 
This project defines several Convolutional Neural Network architectures for Class-Agnostic counting.

## Abstract
In the field of object counting using computer vision techniques, multiple research projects developed highly performant models for counting specific classes. Nonetheless, none has achieved a deep learning model able to count any object independently of the class and very few have worked in that direction. We believe that achieving this class-agnostic counting model is possible by using deep learning methodologies such as Convolutional Neural Networks.

We have developed a set of Convolutional neural networks following different counting strategies to validate our hypothesis and achieve a class-agnostic counting model, as well as to propose the best strategy to achieve a model with these characteristics. The developed models, try several ways of approaching the counting task using common CNN architectures that work well in other environments and by analyzing the counting task from a human point of view while trying to mimic our image processing behavior. Furthermore, three datasets have been defined following different counting approaches for training and evaluating the developed models, even in unseen data to evaluate their generalization capabilities without requiring to be trained on it.

This master thesis proposes what we believe is the best approach for achieving class-agnostic counting as well as a new model improving the current state of the art in class-agnostic counting. We believe that the best approach is by developing a regression-based model with a spatial density output which splits the several tasks that take part in the counting problem into smaller and, if possible, supervised modules specialized in each sub-task. Moreover, this project offers a set of datasets that can be used in future works, fomenting this way the further research of a true class-agnostic counting model which doesn't require to be trained in unseen instances.

## Datasets

The datasets developed in this project are:
 - **Count CIFAR10:** A simple dataset for direct counting where the input is an image grid formed with samples from the CIFAR10 dataset and a template as the reference of the object to be counted.
 
 <p align="center"><img src="./img/Count_CIFAR10.PNG" /></p>
 
 - **ILSVRC2015:** A transformation of the dataset from the ImageNet challenge to fit a spatial density counting task.

 <p align="center"><img src="./img/ILSVRC2015.PNG" /></p>
 
 - **CARPK:** A transformation of the CARPK dataset to fit a spatial density counting task.
 
 <p align="center"><img src="./img/CARPK.PNG" /></p>
 
 ## Spatial density results

Spatial density results for each tested model where the white areas represent the regions where the features of the template were detected, and the red crosses are the local maximum counts.

<p align="center"><img width="500" src="./img/ILSVRC_predictions332.png" /></p>

As can be seen, not all evaluated models achieved good results, however, <b>this project improved the state of the art thanks to the Siamese GMN network, which not only reduced the number of learnable parameters by almost half but also achieved a slightly better object detection.</b>