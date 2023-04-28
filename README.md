# Deep-Learning-Group-75

<!-- PROJECT LOGO -->

<h3 align="center">AEGNN Paper Reproducibility Blog Post</h3>

  <p align="center">
    Group Members: <br>
    David Ninfa (4488040) <br>
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
  The scope of this reproducibility project is limited to reproducing the results from the object recognition task using N-Caltech-101, as shown in table 1 in the AEGNN paper. Furthermore, we carry out hyperparameter tuning and test on a different dataset (N-Cars). For this project we also simplified the method in order for it to be more feasible by disregarding the asynchronous aspect of the AEGNN method with limited computational power. 
  
  
## Datasets
  In the AEGNN paper the N-Caltech-101 and N-Cars dataset is used to compare the proposed AEGNN method to other existing methods for object recognition.
  
### N-Caltech-101
  The Neuromorphic Caltech-101 (N-Caltech) dataset, which was created by Orchard G, et. al (2015) is an event-based version of the original static image dataset Caltech-101. The Caltech-101 dataset is a commonly used benchmark dataset used in computer vision tasks. The N-Caltech-101 dataset contains event streams recorded using an event camera with 101 object categories such as butterfly and umbrella in 8246 event sequences that have a duration of 300ms. Cross-entropy loss and a batch-size of 16 was used for training this dataset in the AEGNN paper. [[2]](#2) 
  
### N-Cars
The Neuromorphic Cars (N-Cars) dataset is an event-based dataset for car classification that contains real events, which are 12,336 car samples and 11,693 non-car samples (background). These events have 24,029 event sequences each of which had a duration of 100ms. [[3]](#3) During training the AEGNN paper used cross-entropy loss and a batch-size of 64 for the N-Cars dataset. [[2]](#2) 
  
  ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/N-Caltech%20and%20N-Cars.png)
  Figure 1: b) N-Caltech dataset, c) N-Cars dataset

## Data pre-processing
  N-Caltech-101 contains binary files with node coordinates.  Figure 2: 3D coordinates in the binary file of umbrella projected in 2D. 
  ![alt text](  https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/umbrella_bin.png)
  
  Elements: 

#### Subsample
  The binary file contains large number of nodes/events. Limit the number of nodes used in the training to a fixed number using fixed point method. 
 
#### Generate graph(edge index)
  Connect each nodes with edge_index from torch_geometric
  
#### Create Edge attributes(Cartesian)
Add Cartesian coordinates of linked nodes in their edge attributes 

  
  


## Baseline Model
  ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/Graph%20res.png)
### Graph res
  The ```GraphRes``` class in the AEGNN repository is the Graph Neural Network used to process the events as spatio-temporal graphs. The neural network has 7 convolution layers and after each convolutional layer there is a batch normalization layer. After this it has a max pooling layer and a fully connected layer. The forward function is also implemented in the GraphRes class and uses the an elu activation function between the layers. This is also depicted in the figure above and more detail can be found in their git repository. 
  
### Recognition.py
  The ```RecognitionModel``` class has the ```GraphRes``` class implemented into it as shown in our code where we define rm: <br>
  ```rm = RecognitionModel(network="graph_res", dataset="ncaltech101", num_classes=NUM_CLASSES, img_shape=(240,180)).to(device)```
  
## Training procedure
  The training procedure as shown below shows that we use the recognition model defined here as model from recognition.py in order to train. The loss criterion is defined as cross entropy loss and the optimizer used is Adam, with a learning rate of 0.1.
  
  ```
  criterion = torch.nn.CrossEntropyLoss().cuda()
  rm = RecognitionModel(network="graph_res", dataset="ncaltech101", num_classes=NUM_CLASSES, img_shape=(240,180)).to(device)
  optimizer = torch.optim.Adam(rm.parameters(), lr = 0.1)
  
  def training(model, data):
  seen = 0
  correct = 0
  
  for item in iter(data):
      item = item.to(device)
      optimizer.zero_grad()
      out = model.forward(item)
      loss = criterion(out, item.y)
      loss.backward()
      pred = out.max(dim=1)[1]
      seen += len(item)
      correct += pred.eq(item.y).sum().item()
      optimizer.step()

  return correct / seen
  ```
  
## Hyperparameters

  
  |           Hyperparameters          | N_samples | N_Classes | N_Epochs | Batch size |
|:----------------------------------:|:---------:|:---------:|:--------:|:----------:|
|          Original Authors          |     25000 |       101 |  Unknown |         16 |
| (Simplified) Reproduction Baseline |       100 |        10 |       15 |         16 |


## Alternative Datasets
  As an extra criteria we tried to implement the AEGNN method on another dataset N-Cars to compare with N-Caltech. 
  
## Results
   A description of the computing infrastructure used
  
  
  
  ### Hyperparameter tuning: N samples

  | Nsamples | time(s)/epoch | train acc | test accuracy |
|----------|---------------|-----------|:-------------:|
| 100      | 3.82s         |       49% |           48% |
| 1000     | 7.4s          |       51% |           39% |
| 5000     | 96.78s        |       37% |           35% |
  
  ### Hyperparameter tuning: N classes
    Test accuracy is relatively lower than train accuracy under all settings. 
classes : umbrella, wheelchair, butterfly… 
Doubling the number of classes resulted in a 50% increase in training time while the performance metric showed minor changes. Under very small output class size of 10 the gap between train and test was the smallest. 

  
  | Nclasses | time(s)/epoch | train acc | test accuracy |
|----------|---------------|-----------|:-------------:|
| 10       | 2.4           |       49% |           48% |
| 50       | 9.9           |       69% |           17% |
| 101      | 15.3          |       69% |           12% |
  
  ### Hyperparameter tuning: N Epochs
After increasing number of epochs to 25, the model shows overfitting behavior in the test. This is due to the high learning rate. In this project, the scope focused on scaled-down, efficient modelling. Thus high learning rate was initially implemented for brief tests whereas the authors of the article had initial lr of 5e-3. 
* Training accuracy *
    ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/nepoch_test.PNG)
 * Test accuracy *
    ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/nepochs_train.PNG)
  

  ### Hyperparameter tuning: Batch size
  Theoretically, a smaller batch size should result in a noisier gradient that might lead to less stable training. However, the test runs showed that the batch size of 8 shows a 5% higher test accuracy. 
  
  | Batch Size | test accuracy |
|------------|:-------------:|
| 16(Base)   |           48% |
| 8          |           53% |
| 4          |           23% |
  
  
## Conclusion
  
## Discussion
  
  ### checklist
  A clear description of the setting, algorithm and models were included in the papaer. However, most assumptions are not identified in the article but in the github repository. The authors used popular datasets thus the dataset themselves had clear statistics and explanation. The article mentions the details of splits however the pre-processing steps were vaguly explained. The code on the repository was outdated and had a few bugs. As it was outdated, it was challenging to set the correct environment and establish dependencies. We were able to find a forked repositor ycontaining old scripts for traning and evaluation however had to redesign the training and evaluation framework again. It was hard to track reasonings behind the authors' selection on the hyper-parameters. 
  
  
  
  
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
