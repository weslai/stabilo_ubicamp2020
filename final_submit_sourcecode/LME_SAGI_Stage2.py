import os
import argparse
import pandas as pd
import torch as t

## import model
from models.cnn_lstm1d import Cnn_LstmNet

## import utils
from pred_data import extract_data

# it should work on window
## get current directory
cwd = os.getcwd()
#MODEL_PATH = 'C:/.../my_model.h5'
## this may should be changed if on windows
MODEL_PATH = cwd + "/checkpoint.ckp"
CLASSES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                'X': 23, 'Y': 24, 'Z': 25, 'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33,
               'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41, 'q': 42, 'r': 43, 's': 44,
               't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49, 'y': 50, 'z': 51}
KEY_LIST = list(CLASSES.keys())
VALUE_LIST = list(CLASSES.values())

## Device Configuration
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

def do_pred_cmd_line():
    '''
    Starts the prediction routine from the command line
    '''
    parser = argparse.ArgumentParser(description='Add path to folder')
    parser.add_argument('-p', '--path', help='Input folder path', required=True)
    parser.add_argument('-c', '--calib', help='Calibration file path', required=True)
    args = parser.parse_args()

    do_pred(args.path, args.calib)


def do_pred(input_folder_path, calib_path):
    '''
    Does the predictions for the given folder and the given calibration file
    :param input_folder_path: the absolute path leading to the folder containing the split csv files
    :param calib_path: the path leading to the calibration file
    :return: the formatted output string (path***prediction~~~path***prediction)
    '''

    # gather all csv files of given path
    letter_files = [f.path for f in os.scandir(input_folder_path) if ".csv" in f.path]

    # load the saved model
    model_ckp = t.load(MODEL_PATH, map_location=device)
    model = Cnn_LstmNet(10, 32)
    model.load_state_dict(model_ckp['state_dict'])
    model.to(device)
    model.eval()  ## evaluation

    # prepare the list of predictions
    predictions = []

    # do the prediction for each file
    for letter_file in letter_files:
        path, prediction = single_char_pred(model, letter_file, calib_path)

        predictions.append((path, prediction))

    # build the return string in the right format (path***prediction~~~path***prediction)
    singlePredictions = [path + "***" + prediction for path, prediction in predictions]
    result_string = ('~~~'.join(singlePredictions))

    # print the result string (this output string will be evaluated by STABILO)
    print(result_string)

    # return only for testing purposes
    return result_string


def single_char_pred(model, letter_path, calib_path):
    '''
    Performs the prediction for a single given letter on the given model. Also takes the calibration into account.
    :param model: the model or None to initialize it for the first time
    :param letter_path: the path to the letter csv file to predict
    :param calib_path: the path to the calibration file
    :return: a (path,prediction) tuple
    '''

    # check if the model was already loaded. Make sure you do not reload the model for each prediction
    if model is None:
        model_ckp = t.load(MODEL_PATH, 'cuda' if t.cuda.is_available() else None)
        model = Cnn_LstmNet(10, 32)
        model.load_state_dict(model_ckp['state_dict'])
        model.to(device)
        model.eval()  ## evaluation

    # read the csv files
    letter = pd.read_csv(letter_path, delimiter=';', index_col=0).drop(columns=["Time"])
    calib = pd.read_csv(calib_path, delimiter=':', header=None, index_col=0).T

    # apply custom preprocessing routines
    letter = do_preprocessing(letter, calib)
    torch_letter = t.tensor(letter)
    torch_letter = t.reshape(torch_letter, (1, torch_letter.shape[0], torch_letter.shape[1]))
    # perform the prediction
    with t.no_grad():
        # output = model([letter])    # output could be [22]
        output = model(torch_letter.to(device, dtype=t.float))
        ## the argmax of the probability
        output = t.argmax(output, dim=1)

    # decode model output
    prediction = KEY_LIST[VALUE_LIST.index(output.cpu()[0])]
    # prediction = CLASSES[int(output.cpu()[0])] #  [22] is 'W'

    # return a (path,prediction) tuple
    return (letter_path, prediction)


def do_preprocessing(letter, calib):
    ## letter is for each file
    ## calib is for each person
    preprocessed_letter = extract_data(letter, calib)
    return preprocessed_letter

# for compiling as executable:
do_pred_cmd_line()
