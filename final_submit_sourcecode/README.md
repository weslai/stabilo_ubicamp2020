# Submission 
## How to run the code

### Installation
1. Create an python Environment \
`conda create --name stabilo_challenge_env python=3.7`\
`conda activate stabilo_challenge_env`
2. Install pytorch \
Install the correct pytorch version according to [official pytorch website](pytorch.org).
3. Install python packages \
`pip install -r requirements.txt`

### Prediction 
run the `LME_SAGI_Stage2.py`\
`-p` for the path of the input data folder\
`-c` for the calibration file path 

### Deep Learning Model
our final trained model is `checkpoint.ckp`, which stored the weights of the our model. \
The final model architecture is under folder models. 
In the folder ```source_code```, there are the codes, which we used to process, train and test on data and our models.
