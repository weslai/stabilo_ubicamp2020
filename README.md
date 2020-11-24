# Stabilo Ubicomp Challenge 2020
This was a challenge, which was hold by the University of Erlangen's MaD-Lab and the Fraunhofer IIS in Nuremberg.\
I did this project during my master study in Summer 2020. \
The task was to classify the handwriting sensor data into 52 Classes(Upper and lower case letters). The handwriting sensor data were recorded of two 3D accelerometers, a 3D gyroscope, a 3D magnetometer and
a 1D force sensor sampled at 100 Hz. \
Our final result got the accuracy for 64.59% in classifying 52 Letters\ 
For more detail, please look into [Stabilo-Challenge](https://stabilodigital.com/competition-details/https://stabilodigital.com/competition-details/)

## Source Code
it can be found under final_submit_sourcecode directory. 
A description can also be found in the directory.
The models are coded with Torch. 

## Short written report 
LME_Stabilo_Challenge_final.pdf
In the report we described our model architecture(CNN + LSTM) and our data processing 
we separated the whole dataset with 80% for training set, and 20% for test set. The writers didn't overlap in training set and test set. This ensured that we preserve the writers' uniqueness in our dataset. 
Each sample had a different length. Before we entered the data into model, we needed to process them to the same length because of the input length of the CNN. Therefore, we used the Fourier method to transcribe each signal to the same length. 
We applied a logarithm scaling for data processing, as we found that logarithmic transformation of the feature space, which results in smaller value ranges being resolved more finely than large ones. 

### Conclusion 
Our proposed model could classify correctly most of the letters from the dataset. Nevertheless, if the upper and lower case letters are too similar, our model wouldn't be able to recognize the correct letter. For example, letters like c, C or o, O. It's sometimes even not easy for human beings to distinguish the difference, as human beings look normally the whole sentense but not just a single letter. 
Also, our model got easily overfitting, and this problem can be solved in the future if we have more data and also more diverse writers. 
There were two improvements on our model during the training process. First, we changed the unit of gyroscope features from angles to radian. Seconds, we added a recurrent neural network(LSTM) after the convolutional neural network. Each of them made the improvement 3-6% for accuracy on test set. 


