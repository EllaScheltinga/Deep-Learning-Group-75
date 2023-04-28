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
  The scope of this reproducibility project is limited to reproducing the results from the object recognition task using N-Caltech-101, as shown in table 1 in the AEGNN paper. Furthermore, we carry out hyperparameter tuning and test on a different dataset (N-Cars).
  
  
## Datasets
  In the AEGNN paper the N-Caltech-101 and N-Cars dataset is used to compare the proposed AEGNN method to other existing methods for object recognition.
  
### N-Caltech-101
  The Neuromorphic Caltech-101 (N-Caltech) dataset, which was created by Orchard G, et. al (2015) is an event-based version of the original static image dataset Caltech-101. The Caltech-101 dataset is a commonly used benchmark dataset used in computer vision tasks. The N-Caltech-101 dataset contains event streams recorded using an event camera with 101 object categories such as butterfly and umbrella in 8246 event sequences that have a duration of 300ms. Cross-entropy loss and a batch-size of 16 was used for training this dataset in the AEGNN paper. [[2]](#2) 
  
### N-Cars
The Neuromorphic Cars (N-Cars) dataset is an event-based dataset for car classification that contains real events, which are 12,336 car samples and 11,693 non-car samples (background). These events have 24,029 event sequences each of which had a duration of 100ms. [[3]](#3) During training the AEGNN paper used cross-entropy loss and a batch-size of 64 for the N-Cars dataset. [[2]](#2) 
  
  ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/N-Caltech%20and%20N-Cars.png)
  Figure 1: b) N-Caltech dataset, c) N-Cars dataset

## Data pre-processing
  

## Baseline Model
  ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/Graph%20res.png)
### Graph res
  The ```GraphRes``` class in the AEGNN repository is the Graph Neural Network used to process the events as spatio-temporal graphs. The neural network has 7 convolution layers and after each convolutional layer there is a batch normalization layer. After this it has a max pooling layer and a fully connected layer. The forward function is also implemented in the GraphRes class and uses the an elu activation function between the layers. This is also depicted in the figure above and more detail can be found in their git repository. 
  
### Recognition.py
  The ```RecognitionModel``` class has the ```GraphRes``` class implemented into it as shown in our code where we define rm: <br>
  ```rm = RecognitionModel(network="graph_res", dataset="ncaltech101", num_classes=NUM_CLASSES, img_shape=(240,180)).to(device)```
  
## Training procedure
  
  
## Hyperparameters
  |           Hyperparameters          | N_samples | N_Classes | N_Epochs | Batch size |
|:----------------------------------:|:---------:|:---------:|:--------:|:----------:|
|          Original Authors          |     25000 |       101 |  Unknown |         16 |
| (Simplified) Reproduction Baseline |       100 |        10 |       15 |         16 |
  
## Epochs and sample size
  
## Alternative Datasets
  
## Results
  
## Conclusion
  
## Discussion
  
## References
  (N-Caltech) <br>
  <a id="1">[1]</a> 
  Orchard, G., Jayawant, A., Cohen, G. H., & Thakor, N. V. (2015). Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades. Frontiers in Neuroscience, 9. https://doi.org/10.3389/fnins.2015.00437
  
  (AEGNN) <br>
  <a id="2">[2]</a> 
  Schaefer, S. M., Gehrig, D., & Scaramuzza, D. (2022). AEGNN: Asynchronous Event-based Graph Neural Networks. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2203.17149
  
  (N-Cars) <br>
  <a id="3">[3]</a> 
  Sironi, A., Brambilla, M., Bourdis, N., Lagorce, X., & Benosman, R. (2018). HATS: Histograms of Averaged Time Surfaces for Robust Event-based Object   Classification. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1803.07913
