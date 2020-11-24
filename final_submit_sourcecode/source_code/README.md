# Documentation 

## Preprocessing 
Three main files for preprocessing are ```merge_dataset.py```, ```load_whole_data.py```, and ```utils_cnn.py```

The file ```merge_dataset.py``` loads the data from the given path, extracts data, does the calibration.\
The file ```load_whole_data.py``` it does the main preprocessing, which gets rid of magnetic features, transforms the unit from angle to radian in gyroscope feature, and also resample the data into the same length using Fourier method.\
Then, it splits the data into training and test set in ```merge_dataset.py```, based on the percentage 80 and 20 as default.\
All the functions are called inside ```train.py``` (main).\
One can start modify from ```train.py```.

For ```PadCollate.py```, which was used for Encoder Decoder model because of different lengths of the input data. However, it didn't work out really well in our case. 
## Training
We trained on 80% of the dataset. \
```train.py```is the main file, which is used to train the model. All models are saved under directory ```/model```, the one we submitted is ```cnn_lstm1d.py```.\ 
Under directory ```/model```, one can also find some models, like Encoder LSTM or ResNet with LSTM, which we tried on the dataset. 

```train.py``` will eventually call ```trainer.py```, which does the main job of training and evaluating the result, and it saves losses and accuracies. 
Later on, it gives back to ```train.py``` to plot the result for visualization.\ 
one can change the data path in ```train.py``` to train another dataset.

## Testing(Evaluation) 
We test on 20% of the dataset. \
For model testing, it should be done with the files ```LME_SAGI_Stage2.py``` and ```pred_data.py```, they will load the model and a checkpoint to evaluate on the data file with a given path and corresponding calibration file.
