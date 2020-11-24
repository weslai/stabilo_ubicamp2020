from sklearn.model_selection import train_test_split
from utils_cnn import zeromean_normalize, zeromean_normal_oneperson
from load_whole_data import read_data_to_dataset

COLUMNS = {'Acc1 X', 'Acc1 Y', 'Acc1 Z', 'Acc2 X', 'Acc2 Y', 'Acc2 Z',
               'Gyro X', 'Gyro Y', 'Gyro Z', 'Mag X', 'Mag Y', 'Mag Z'}
LETTER_DICT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                'X': 23, 'Y': 24, 'Z': 25, 'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33,
               'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41, 'q': 42, 'r': 43, 's': 44,
               't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49, 'y': 50, 'z': 51, '0': 53, '*': 52} ## '0' blank endtoken, '*' starttoken

def read_rawdata_to_person(file_path, all_data= True, millisec=1000, resample=True): ## file_path 'str'
    ## this function make the rawdata into the same length of data
    ## return person in List
    person_tuples = read_data_to_dataset(file_path, all_data, millisec=millisec, resample=resample)
    return person_tuples

def data_split(data, alpha=0.2):
    """alpha for the split radio"""
    train_list = []
    test_list = []
    dataset = []
    train_data, test_data = train_test_split(data, test_size=alpha, random_state=42, shuffle=True)
    for trains in train_data:
        for i in range(len(trains)):
            train_list.append((trains[i][0], trains[i][1]))
    for tests in test_data:
        for i in range(len(tests)):
            test_list.append((tests[i][0], tests[i][1]))
    dataset.append((train_list, test_list))
    return dataset

def zero_mean_person(persons, all_data= True):
    train_list = []
    test_list = []

    alpha = 0.2
    train_data, test_data = train_test_split(persons, test_size=alpha, random_state=42, shuffle=True)
    train_data, test_data = zeromean_normalize(train_data, test_data)

    for trains in train_data:
        for i in range(len(trains)):
            train_list.append((trains[i][0], trains[i][1]))
            ## for padding into dataloader, we want to know the original data length
            # train_len.append(trains[0][i].shape[0])
    for tests in test_data:
        for i in range(len(tests)):
            test_list.append((tests[i][0], tests[i][1]))
            ## for padding into dataloader, we want to know the original data length
            # test_len.append(tests[0][i].shape[0])
    ## increasing index
    # train_idx = sorted(range(len(train_list)), key=lambda k: train_list[k][0].shape[0])
    # test_idx = sorted(range(len(test_list)), key=lambda k: test_list[k][0].shape[0])
    ## make it decreasing
    # train_idx = train_idx[::-1]
    # test_idx = test_idx[::-1]

    zeromean_dataset = []
    zeromean_dataset.append((train_list, test_list))
    return zeromean_dataset
#pickle.dump(new_person, open(file_path + 'stage2_plus500.pkl', 'wb')) ## add a half second


def zero_mean_one_person(persons):
    train_list = []
    test_list = []

    data = persons[0][0]
    label = persons[0][1]
    alpha = 0.2
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=alpha, random_state=42, shuffle=True)
    train_data, test_data = zeromean_normal_oneperson(train_data, test_data)
    for trains, label in zip(train_data, train_labels):
        train_list.append((trains, label))
            ## for padding into dataloader, we want to know the original data length
        # train_len.append(trains.shape[0])
    for tests, label in zip(test_data, test_labels):
        test_list.append((tests, label))
            ## for padding into dataloader, we want to know the original data length
        # test_len.append(tests.shape[0])
    ## increasing index
    # train_idx = sorted(range(len(train_list)), key=lambda k: train_list[k][0].shape[0])
    # test_idx = sorted(range(len(test_list)), key=lambda k: test_list[k][0].shape[0])
    ## make it decreasing
    # train_idx = train_idx[::-1]
    # test_idx = test_idx[::-1]

    zeromean_dataset = []
    zeromean_dataset.append((train_list, test_list))
    return zeromean_dataset
