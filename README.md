# Deep-Learning-Group-75

<!-- PROJECT LOGO -->

<h3 align="center">AEGNN Paper Reproducibility Blog Post</h3>

  <p align="center">
    Group Members: <br>
    David Ninfa (4488040) <br>
    Ella Scheltinga (4833856) <br>
    Mia Choi (5401321) 
  
## Overview of Paper
  The AEGNN paper proposes a novel method for event-processing using Asynchronous Event-based Graph Neural Networks. Graph Neural Networks process events as static spatio-temporal graphs, which are sparsely distributed. Introducing Asynchronous Event-based Graph Neural Networks they aim to process events as evolving spatio-temporal graphs. The AEGNN method can efficiently update because the recomputation of the network activations is restricted to only the nodes affected by a new event. The AEGNN paper validates its method using object recognition and detection tasks. In this paper, object recognition pertains to predicting an object class from the event stream and object detection refers to classifying and detecting object bounding boxes from an event stream. The results from the paper show a significant reduction in the computational complexity and computational latency.
  
  
## Scope of reproducibility
  The scope of this reproducibility project is limited to reproducing the results from the object recognition task using N-Caltech-101, as shown in table 1 in the AEGNN paper. Furthermore, we carry out hyperparameter tuning and test on a different dataset (N-Cars). For this project we also simplified the method in order for it to be more feasible by disregarding the asynchronous aspect of the AEGNN method with limited computational power. 
  
  
## Dataset
  In the AEGNN paper the N-Caltech-101 and N-Cars dataset is used to compare the proposed AEGNN method to other existing methods for object recognition. In order to use the datasets below please download them by using the link provided and make a new folder in the repository called data and add them here. 
  
### N-Caltech-101
  The Neuromorphic Caltech-101 (N-Caltech) dataset, which was created by Orchard G, et. al (2015) is an event-based version of the original static image dataset Caltech-101. The Caltech-101 dataset is a commonly used benchmark dataset used in computer vision tasks. The N-Caltech-101 dataset contains event streams recorded using an event camera with 101 object categories such as butterfly and umbrella in 8246 event sequences that have a duration of 300ms. Cross-entropy loss and a batch-size of 16 was used for training this dataset in the AEGNN paper. [[2]](#2) 
  
### N-Cars
The Neuromorphic Cars (N-Cars) dataset is an event-based dataset for car classification that contains real events, which are 12,336 car samples and 11,693 non-car samples (background). These events have 24,029 event sequences each of which had a duration of 100ms. [[3]](#3) During training the AEGNN paper used cross-entropy loss and a batch-size of 64 for the N-Cars dataset. [[2]](#2) 
  
  ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/N-Caltech%20and%20N-Cars.png)
  Figure 1: b) N-Caltech dataset, c) N-Cars dataset

## Data pre-processing
  N-Caltech-101 contains binary files with node coordinates.  Figure 2: 3D coordinates in the binary file of umbrella projected in 2D. 
  ![alt text](  https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/umbrella_bin.png)
 
#### Subsampling
  The binary file contains large number of nodes/events. Limit the number of nodes used in the training to a fixed number using fixed point method. 
 
#### Generating graph
  Connect each nodes with edge_index from torch_geometric
  
#### Creating edge attibutes
Add Cartesian coordinates of linked nodes in their edge attributes 

## Baseline Model
  ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/Graph%20res.png)
### GraphRes
The ```GraphRes``` class in the AEGNN repository is the Graph Neural Network used to process the events as spatio-temporal graphs. The neural network has seven convolution layers each followed by a batch normalization layer. After this there is a max pooling layer and a fully connected layer. The forward function is also implemented in the GraphRes class and uses the an elu activation function between the layers as depicted in the figure above. More detail can be found in the author's git repository. 

 
### RecognitionModel
  ```RecognitionModel``` is the umbrella of all code with the aim in this case to correctly identify an object.
  This class implements ```GraphRes``` network as follows: <br>
  ```rm = RecognitionModel(network="graph_res", dataset="ncaltech101", num_classes=NUM_CLASSES, img_shape=(240,180)).to(device)```
   This class implements ```GraphRes``` to construct the networked graphs and incorporates Cross Entrepoy as loss function.
  
## Training procedure
 The training procedure as shown below shows that we use the recognition model defined here as model from recognition.py in order to train. The hyperparameters has been scaled down for an affordable computation under basic Google Colab python notebook setting: 12.7 GB System RAM, Google colab GPU RAM with CUDA. The loss criterion is defined as cross entropy loss and the optimizer used is Adam, with a learning rate of 0.1. The loss and adam were not explicitly mentioned in the code because there was no training function present in the code. However, it was mentioned in the paper. The authors implemented decreasing learning rate yet for the sake of simplicity constant learning rate was implemented. Apply the model for the object recognition task. 

  
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
|          Original Authors          |     25000 |       101 |  20+ (slow decay)  |         16 |
| (Simplified) Reproduction Baseline |       100 |        10 |       15 (constant) |         16 |


## Alternative Dataset: N-Cars
 As an extra criteria we tried to implement the AEGNN method on another dataset N-Cars to compare with N-Caltech. The data was encoded in a binary file of the type .tar.gz. for which there was no code to read the data, however this was easy to implement. The dataset was already split into a train, test and validation set. The pre-processing steps were very similar compared to the N-Caltech-101 dataset, therefore required minimal time. There was code provided to read the labels of the data, although there was little code implementation or guidance given on how to add labels to the given data sequences in order to train the data using these labels. The important differnce with the NCaltech101 dataset is that problem is a multi-class problem as with this dataset, the classifactio is binary: a even stream is showing a car (label "1" or True) or whethet it is showing a background without a car (label "0", or _ False_).
  
For both the training batch and the validation (test) batch, 500 event streams are used, divided over 10 batches with 15 epochs, as can be seen below. The training proces ended in a 63% accurate objec recognition network, where the validation was left with only 49%. This is significantly lower than what the network performed in the performance of the network with the N-Cars dataset in the AEGNN paper where the accuracy was 0.945.
  
  Overall, debugging took longer with this dataset because every time you had to restart the kernel in the google collab environment you had to load the entire dataset and this takes around 10mins. Only after this can you scale down the dataset in order to debug faster. Even after this scaling down step the ```pre_transform_all``` function takes very long. Furthermore, a lot of code that was used for N-Caltech-101 was transferable to N-Cars and the basic code given for reading and loading the code was also provided. 
  
 ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/WhatsApp%20Image%202023-04-28%20at%2023.05.26.jpeg)
 ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/WhatsApp%20Image%202023-04-28%20at%2023.07.57.jpeg)
  
# Experiment: Hyperparameter tuning on NCaltech101
  
  ### Hyperparameter tuning Results: N samples

  Amount of nodes to be sampled from each timeframe, fixed. 
  
  | Nsamples | time(s)/epoch | train acc | test accuracy |
|----------|---------------|-----------|:-------------:|
| 100      | 3.82s         |       49% |           48% |
| 1000     | 7.4s          |       51% |           39% |
| 5000     | 96.78s        |       37% |           35% |
  
  ### Hyperparameter tuning Results: N classes
Test accuracy is relatively lower than train accuracy under all settings. 
classes : umbrella, wheelchair, butterfly… 
Doubling the number of classes resulted in a 50% increase in training time while the performance metric showed minor changes. Under very small output class size of 10 the gap between train and test was the smallest. 

  
  | Nclasses | time(s)/epoch | train acc | test accuracy |
|----------|---------------|-----------|:-------------:|
| 10       | 2.4           |       49% |           48% |
| 50       | 9.9           |       69% |           17% |
| 101      | 15.3          |       69% |           12% |
  
  ### Hyperparameter tuning Results: N Epochs
After increasing number of epochs to 25, the model shows overfitting behavior in the test. This is due to the high learning rate. In this project, the scope focused on scaled-down, efficient modelling. Thus high learning rate was initially implemented for brief tests whereas the authors of the article had initial lr of 5e-3. 
  | N Epoch | train acc | test accuracy |
|---------|-----------|:-------------:|
| 15      |       49% |           48% |
| 25      |      100% |          100% |
| 35      |      100% |          100% |
  
* Training accuracy *
    ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/nepoch_test.PNG)
 * Test accuracy *
    ![alt text](https://github.com/EllaScheltinga/Deep-Learning-Group-75/blob/main/nepochs_train.PNG)
  

  ### Hyperparameter tuning Results: Batch size
  Theoretically, a smaller batch size should result in a noisier gradient that might lead to less stable training. However, the test runs showed that the batch size of 8 shows a 5% higher test accuracy. 
  
  | Batch Size | test accuracy |
|------------|:-------------:|
| 16(Base)   |           48% |
| 8          |           53% |
| 4          |           23% |
  
## Discussion
  A clear description of the setting, algorithm and models were included in the papaer. However, most assumptions are not identified in the article but in the github repository. The authors used popular datasets thus the dataset themselves had clear statistics and explanation. The article mentions the details of splits however the pre-processing steps were vaguly explained. The code on the repository was outdated and had a few bugs, especially in pre-processing step. Some of the libraries were outdated and it was challenging to set the correct environment and establish dependencies. We were able to find a forked repositor ycontaining old scripts for traning and evaluation however had to redesign the training and evaluation framework again. It was hard to track reasonings behind the authors' selection on the hyper-parameters

  
   #### Simplification
 Disregarding Asynchronous
 Asynchronousness is the strength of graph data compared to other types of NNs. However it was not possible to explore this propertity due to limited time and resources.
  
## Conclusion
  The goal of this project was to reproduce the object recognition task with the N-Caltech-101 dataset using a Graph Neural Network as described in the AEGNN paper. The input here are event streams: these are thinned out, processed and converted to graphs where a neural network will then learn to connect the core markers for a specific classification. This method is especially interesting because of the smaller data quantities and computational power required compared to traditional images and associated neural networks. 
  
  Due to unclarity and errors in the original article and the code and limited computational resources, the original claim of 60% accuracy on the N-Caltech101 dataset was not reproducable. The highest accuracy of 53% was reached with 100 samples, 10 classes, 15 epochs and 8 batches and with high learning rate of 0.1%. For further implementation, it is recommended to secure computational power through cloud computing and further tune learning rate to prevent overfitting that we caught in epoch tuning. 
  
  The reproducibility using N-Cars was succeful to an extent as the preprocessed data could be trained and the result is an accuracy 0.49 for the object detection task. Compared to the 0.945 accuracy stated in the AEGNN paper this is significanlty lower, however due to the resources availale there were some limitations, which could have lead to a lower resutl. Overall, most of the building blocks of code to reproduce N-Cars are present, however there are some missing parts which make reproducing difficult.
 
  
## Future research
  - Adding the asynchronous property to the event-based Graph Neural Network


  
## Contributions
- Mia Choi: Simplification, Implementation, experiment, blog post 
- David Ninfa: Simplification, Implementation, debugging, blog post 
- Ella Scheltinga: Simplification, Implementation, debugging N-Cars dataset, blog post 

  
## References
  <a id="1">[1]</a> 
  Orchard, G., Jayawant, A., Cohen, G. H., & Thakor, N. V. (2015). Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades. Frontiers in Neuroscience, 9. https://doi.org/10.3389/fnins.2015.00437

  <a id="2">[2]</a> 
  Schaefer, S. M., Gehrig, D., & Scaramuzza, D. (2022). AEGNN: Asynchronous Event-based Graph Neural Networks. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2203.17149
  
  <a id="3">[3]</a> 
  Sironi, A., Brambilla, M., Bourdis, N., Lagorce, X., & Benosman, R. (2018). HATS: Histograms of Averaged Time Surfaces for Robust Event-based Object   Classification. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1803.07913
