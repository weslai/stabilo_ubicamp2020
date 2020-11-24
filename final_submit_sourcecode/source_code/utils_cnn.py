from typing import List
import numpy as np

def zeromean_normalize(Traindata: List, Testdata: List):
    """ (Used in Stage 2) Sequence data of letter
     Returns zero-mean normalized Data for the whole train_data,
    and save the mean and conv of train for testdata """
    data_array = None
    for train_data in Traindata:  ## writer
        for sample in train_data: ## sample in each writer
            if data_array is None:
                data_array = sample[0].to_numpy()
                # data_array = sample.loc[:0].to_numpy()
                # data_array = data_array[:-1, :]
            else:
                # data_array = np.concatenate((data_array, sample.loc[:0].to_numpy()[:-1, :]), axis= 0)
                data_array = np.concatenate((data_array, sample[0].to_numpy()), axis= 0)   ## concatenate the sample in each writer
    train_means = np.mean(data_array, axis= 0)
    train_stds = np.std(data_array, axis= 0)

    for i in range(len(Traindata)):
        for j in range(len(Traindata[i])):
            # data_array = sample.loc[:0].to_numpy()[:-1, :]
            data_array = Traindata[i][j][0].to_numpy()
            data_array = (data_array - train_means) / train_stds  ## normalized
        # sample.loc[:0] = np.concatenate((data_array, np.zeros((1, data_array.shape[1]))), axis= 0)
        # train_data[0][num] = data_array
            Traindata_list = list(Traindata[i][j])
            Traindata_list[0] = data_array
            Traindata[i][j] = tuple(Traindata_list)

    ## in Test Data we also use Train mean and std to normalize
    for i in range(len(Testdata)):
        for j in range(len(Testdata[i])):
            # data_array = sample.loc[:0].to_numpy()[:-1, :]
            data_array = Testdata[i][j][0].to_numpy()
            data_array = (data_array - train_means) / train_stds  ## normalized
            # sample.loc[:0] = np.concatenate((data_array, np.zeros((1, data_array.shape[1]))), axis=0)
            # test_data[0][num] = data_array
            Testdata_list = list(Testdata[i][j])
            Testdata_list[0] = data_array
            Testdata[i][j] = tuple(Testdata_list)
    return Traindata, Testdata
    # zeromeaned = preprocessing.scale(data_array, axis= 0)
    # dict_normalized = {'Acc1 X': zeromeaned[:, 0], 'Acc1 Y': zeromeaned[:, 1], 'Acc1 Z': zeromeaned[:, 2],
    #                    'Acc2 X': zeromeaned[:, 3], 'Acc2 Y': zeromeaned[:, 4], 'Acc2 Z': zeromeaned[:, 5],
    #                    'Gyro X': zeromeaned[:, 6], 'Gyro Y': zeromeaned[:, 7], 'Gyro Z': zeromeaned[:, 8],
    #                    'Mag X': zeromeaned[:, 9], 'Mag Y': zeromeaned[:, 10], 'Mag Z': zeromeaned[:, 11],
    #                    'Force': zeromeaned[:, 12]}
    # zeromean_scaled = pd.DataFrame(dict_normalized)

def zeromean_normal_oneperson(Traindata: List, Testdata: List):
    """ (Used in Stage 2) Sequence data of letter
     Returns zero-mean normalized Data for the one person train_data,
    and save the mean and conv of train for testdata """
    data_array = None
    for train_data in Traindata:  ##sample in each writer
        if data_array is None:
            data_array = train_data.to_numpy()
                # data_array = sample.loc[:0].to_numpy()
                # data_array = data_array[:-1, :]
        else:
                # data_array = np.concatenate((data_array, sample.loc[:0].to_numpy()[:-1, :]), axis= 0)
            data_array = np.concatenate((data_array, train_data.to_numpy()), axis=0)  ## concatenate the sample in each writer
    train_means = np.mean(data_array, axis=0)
    train_stds = np.std(data_array, axis=0)

    for i in range(len(Traindata)):
            # data_array = sample.loc[:0].to_numpy()[:-1, :]
        data_array = Traindata[i].to_numpy()
        data_array = (data_array - train_means) / train_stds  ## normalized
            # sample.loc[:0] = np.concatenate((data_array, np.zeros((1, data_array.shape[1]))), axis= 0)
            # train_data[0][num] = data_array
        Traindata[i] = data_array

    ## in Test Data we also use Train mean and std to normalize
    for i in range(len(Testdata)):
            # data_array = sample.loc[:0].to_numpy()[:-1, :]
        data_array = Testdata[i].to_numpy()
        data_array = (data_array - train_means) / train_stds  ## normalized
            # sample.loc[:0] = np.concatenate((data_array, np.zeros((1, data_array.shape[1]))), axis=0)
            # test_data[0][num] = data_array
        Testdata[i] = data_array
    return Traindata, Testdata