# DL-for-sEMG

#### 1.《[Deep Learning with Convolutional Neural Networks Applied to Electromyography Data: A Resource for the Classification of Movements for Prosthetic Hands](https://www.frontiersin.org/articles/10.3389/fnbot.2016.00009/full)》

#### Journal: [frontiers in Neurorobotics](https://www.frontiersin.org/journals/neurorobotics)-----2016

**Abstract**—Natural control methods based on surface electromyography (sEMG) and pattern recognition are promising for hand prosthetics. However, the control robustness offered by scientific research is still not sufficient for many real life applications, and commercial prostheses are capable of offering natural control for only a few movements. In recent years deep learning revolutionized several fields of machine learning, including computer vision and speech recognition. Our objective is to test its methods for natural control of robotic hands via sEMG using a large number of intact subjects and amputees. We tested convolutional networks for the classification of an average of 50 hand movements in 67 intact subjects and 11 transradial amputees. The simple architecture of the neural network allowed to make several tests in order to evaluate the effect of pre-processing, layer architecture, data augmentation and optimization. The classification results are compared with a set of classical classification methods applied on the same datasets. The classification accuracy obtained with convolutional neural networks using the proposed architecture is higher than the average results obtained with the classical classification methods, but lower than the results obtained with the best reference methods in our tests. The results show that convolutional neural networks with a very simple architecture can produce accurate results comparable to the average classical classification methods. They show that several factors (including pre-processing, the architecture of the net and the optimization parameters) can be fundamental for the analysis of sEMG data. Larger networks can achieve higher accuracy on computer vision and object recognition tasks. This fact suggests that it may be interesting to evaluate if larger networks can increase sEMG classification accuracy too.

![](G:\Typora\DL-for-sEMG\images\1.png)

##### Architecture:

**Input layer**: the input data correspond to time windows of 150 ms, spanning all the electrode

​                     measurements available.              

**BLOCK1**: one convolutional layer (32 filters) + ReLU.

**BLOCK2**: one convolutional layer (32 filters, 3x3) + ReLU + average pooling (3x3).

**BLOCK3**: one convolutional layer (64 filters, 5x5) + ReLU + average pooling (3x3).

**BLOCK4**: one convolutional layer (64 filters, 5x1 for Otto Bock and 9x1 for Delsys) + ReLU.

**BLOCK5**: one convolutional layer (1x1) + softmax layer.

##### Hyper-parameters:

SGD momentum: 0.9.

learning rate: 0.001.

weight decay: 0.0005.

batch size: 256.

epoch: 30.

 ##### Data Augmentation:

data were doubled and white Gaussian noise was added to the new set.

##### Results:

![](G:\Typora\DL-for-sEMG\images\2.png)

###### Dataset 1: 

CNN: 66.59 $\pm$ 6.40%.

Average accuracy using classical methods: 62.06 $\pm$ 6.07%.

Best classification with Random Forests: 75.32 $\pm​$ 5.69%.              

###### Dataset 2:

CNN: 60.27 $\pm$ 7.7%.

Average accuracy using classical methods: 60.28 $\pm$ 6.51%.

Best classification with Random Forests: 75.27 $\pm$ 7.89%.    

###### Dataset 3:

CNN: 38.09 $\pm$ 14.29%.

Average accuracy using classical methods: 38.82 $\pm​$ 11.99%.

Best classification with SVM: 46.27 $\pm​$ 7.89%. 

Nvidia Titan-X GPU, Training time is 1 h and 42 min, testing time is 21.5 s.

##### Comment:

The results show that convolutional neural networks with a very simple architecture can produce accurate results comparable to the average classical classification methods, but lower than the results obtained with the best reference methods in their tests.

In this article, they introduce a baseline for the application of convolutional neural networks to the classification of hand movements by sEMG and they compare the results with a set of classical machine learning methods on a large set of movements and subjects (including also amputees). 



#### 2.《[Self-Recalibrating Surface EMG Pattern Recognition for Neuroprosthesis Control Based on Convolutional Neural Network](https://www.frontiersin.org/articles/10.3389/fnins.2017.00379/full)》

#### Journal: [frontiers in Neuroscience](https://www.frontiersin.org/journals/neuroscience)-----2017

**Abstract**—Hand movement classification based on surface electromyography (sEMG) pattern recognition is a promising approach for upper limb neuroprosthetic control. However, maintaining day-to-day performance is challenged by the non-stationary nature of sEMG in real-life operation. In this study, we propose a self-recalibrating classifier that can be automatically updated to maintain a stable performance over time without the need for user retraining. Our classifier is based on convolutional neural network (CNN) using short latency dimension-reduced sEMG spectrograms as inputs. The pretrained classifier is
recalibrated routinely using a corrected version of the prediction results from recent testing sessions. Our proposed system was evaluated with the NinaPro database comprising of hand movement data of 40 intact and 11 amputee subjects. Our system was able to achieve ∼10.18% (intact, 50 movement types) and ∼2.99% (amputee, 10 movement types) increase in classification accuracy averaged over five testing sessions with respect to the unrecalibrated classifier. When compared with a support vector machine (SVM) classifier, our CNN-based system consistently showed higher absolute performance and larger improvement as well as more efficient training. These results suggest that the proposed system can be a useful tool to facilitate long-term adoption of prosthetics for amputees in real-life applications.

![](G:\Typora\DL-for-sEMG\images\3.png)

##### Preprocessing:

1. sEMG signals are sectioned into 200 ms (400 samples) segments with 100 ms (200 samples) increments.
2. The spectrogram for each segment of each channel is computed using a 256-point fast Fourier transform (FFT) with a Hamming window and 184-point overlap.
3. The intensity of each spectrogram is normalized into 0 to 1.
4. To improve computational efficiency and performance, they vectorize the normalized spectrogram matrices channel by channel and then apply PCA to it.

##### Architecture:

Their CNN model contains  1 convolutional layer, 2 fully connected layers with dropout and a soft max loss layer. The soft max loss layer computes the cost function using the normalized exponential function. Each layer is trained by back propagation. 

Use ReLU as activation function which has been shown to help avoid problem of vanishing gradient. Also apply dropout method to reduce over fitting.

**Environment**: An open source MATLAB toolbox MatConvNet was used to implement the CNN classifier.



![](G:\Typora\DL-for-sEMG\images\4.png)

​                                      Block diagram of the proposed self-recalibration classifier.

##### Results:

![](G:\Typora\DL-for-sEMG\images\5.png)

##### Comment:

They propose a self-recalibrating classifier based on convolutional neural network (CNN) using short latency dimension-reduced sEMG spectrograms as inputs that can be automatically updated to maintain a stable performance over time without the need for user retraining.



#### 3. 《[sEMG-Based Gesture Recognition with Convolution Neural Networks](<https://www.mdpi.com/2071-1050/10/6/1865>)》

#### Journal: [Sustainability](https://www.mdpi.com/journal/sustainability)-----2018

**Abstract**—The traditional classification methods for limb motion recognition based on sEMG have been deeply researched and shown promising results. However, information loss during feature extraction reduces the recognition accuracy. To obtain higher accuracy, the deep learning method was introduced. In this paper, we propose a parallel multiple-scale convolution architecture. Compared with the state-of-art methods, the proposed architecture fully considers the characteristics of the sEMG signal. Larger sizes of kernel filter than commonly used in other CNN-based hand recognition methods are adopted. Meanwhile, the characteristics of the sEMG signal, that is, muscle independence, is considered when designing the architecture. All the classification methods were evaluated on the NinaPro database. The results show that the proposed architecture has the highest recognition accuracy. Furthermore, the results indicate that parallel multiple-scale convolution architecture with larger size of kernel filter and considering muscle independence can significantly increase the classification accuracy.

##### Preprocessing:

![](G:\Typora\DL-for-sEMG\images\6.png)

The sEMG signals were segmented using a sliding window with 100 ms (200 samples). 

The sEMG signals from 12 electrodes were converted into the sEMG images of size 12 * 200.

##### Architecture:

![](G:\Typora\DL-for-sEMG\images\7.png)

In the Block 1, five convolution layers and two maximum pooling layers are employed. The first three convolution layers contain 40 2D filters of 1x13 with the stride of 1 and a zero padding of 0. The last two convolution layers are similar to anterior layers except for the first dimension of kernel filter. In these two layers, the information from different electrodes is mixed to detect the relevance of each electrode. The two maximum pooling layers using the filters of 1x2 are followed by the first and second convolution layer, respectively.

The Block 2 is different in first three convolution layers which adopt the bigger filter kernel size. The first three convolution layers contain 40 2D filters of 1x57 with the stride of 1 and a zero padding of 0. The following two convolution layers are the same as the last two convolution layers of Block 1. The pooling layers were not adopted in Block 2.

The classifier is composed of three fully connected layers and a softmax layer. The input layer consists of 520 units which are corresponding to the feature extracted by two blocks. The first and the second hidden layers consist of 260 and 130 units, respectively. The output layer has 17 units which are equal to the number of hand gestures.

In both blocks, the batch normalization is employed between each convolution layer and activation function. In classifier, after first and second fully connected layers, the dropout with a probability of
0.5 is adopted.

##### Comparisons:

![](G:\Typora\DL-for-sEMG\images\8.png)

By comparing the results of C-SK2, C-2B1, C-2B2, and C-B1PB2, the influence of the different size
of kernel filter on classification accuracy can be obtained. The results of C-DK and C-B1PB2, or C-SK
and C-SK2 can reveal the effect of considering the sEMG signal characteristics on classification accuracy.
Moreover, we evaluated the C-B1PB2 on all hand gestures of NinaPro DB2 (including Exercises B, C
and D) to verify the effectiveness of the proposed classification algorithm.

##### Enviroment:

The Classical Classification method was implemented in MATLAB. The others were implemented
with PyTorch.

##### Results:

![](G:\Typora\DL-for-sEMG\images\9.png)

Table 2 gives the average classification accuracy results for each experiment.

![](G:\Typora\DL-for-sEMG\images\10.png)

##### Comments:

The comparison experiments in this paper are very full, such as kernel size as on. The special feature of this network is has two parallel blocks to extract features and concatenate together.