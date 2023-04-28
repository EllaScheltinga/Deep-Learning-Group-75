# Deep-Learning-Group-75

<!-- PROJECT LOGO -->

<h3 align="center">AEGNN Paper Reproducibility Blog Post</h3>

  <p align="center">
    Group Members: <br>
    David () <br>
    Ella Scheltinga (4833856) <br>
    Mia Choi (5401321) 



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Introduction
  
  
  
## Overview of Paper
  The AEGNN paper proposes a novel method for event-processing using Asynchronous Event-based Graph Neural Networks. Graph Neural Networks process events as static spatio-temporal graphs, which are sparsely distributed. Introducing Asynchronous Event-based Graph Neural Networks they aim to process events as evolving spatio-temporal graphs. The AEGNN method can efficiently update because the recomputation of the network activations is restricted to only the nodes affected by a new event. The AEGNN paper validates its method using object recognition and detection tasks. In this paper, object recognition pertains to predicting an object class from the event stream and object detection refers to classifying and detecting object bounding boxes from an event stream. 
  
  
## Scope of reproducibility
  
## Datasets
  In the AEGNN paper the N-Caltech-101 and N-Cars dataset is used to compare the proposed AEGNN method to other existing methods for object recognition.
  
### N-Caltech-101
  The Neuromorphic Caltech-101 (N-Caltech) dataset, which was created by Orchard G, et. al (2015) is an event-based version of the original static image dataset Caltech-101. The Caltech-101 dataset is a commonly used benchmark dataset used in computer vision tasks. The N-Caltech-101 dataset contains event streams recorded using an event camera with 101 object categories such as butterfly and umbrella in 8246 event sequences that have a duration of 300ms. Cross-entropy loss and a batch-size of 16 was used for training this dataset in the AEGNN paper. (cite AEGNN paper). 
  
### N-Cars
The N-Cars dataset is an event-based dataset for car classification that contains real events, which are 12,336 car samples and 11,693 non-car samples (background). These events have 24,029 event sequences each of which had a duration of 100ms. During training the AEGNN paper used cross-entropy loss and a batch-size of 64 for the N-Cars dataset.

## Data pre-processing
  

## Creating the models
  

## Baseline Model
  
### Graph res
  
### Recognition.py
  
## Training procedure
  
  
## Hyperparameters
  
## Epochs and sample size
  
## Alternative Datasets
  
## Results
  
## Conclusion
  
## Discussion
  
## References
