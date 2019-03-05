from PIL import Image
import numpy as np
import random
import itertools

def get_labels(path='D://Dokumente//Studium//TU Berlin - ITM//HTCV//Random_Ferns//Code_Development//Data//'):
    '''
    :param path: path to the data which contains the labels for the oberpfaffenhoffen dataset.
    :return: matrix where city = 1; field = 2; forest = 3; grassland = 4; street = 5; unlabelled = 0

    This function reads in the labels and returns a labelmatrix
    '''

    label_paths = ['city_inter.png',
                   'field_inter.png',
                   'forest_inter.png',
                   'grassland_inter.png',
                   'street_inter.png']
    # collect labels from all five images specified in label_paths
    size = np.array(Image.open(path+label_paths[0])).shape
    l = np.zeros(size)
    for idx, i in enumerate(label_paths):
        p = path+i
        ci = np.asarray(Image.open(p)).copy()
        ci[ci < 128] = 0
        ci[ci >= 128] = 1
        l += (idx+1)*ci
    return np.asarray(l, dtype = np.int8)


def test_train_split(matrix, patchsize, k=5):
    '''

    :param matrix: np.array of size x,y,3,3 - rat image represented as coherency or covariance matrix
    :param patchsize: int - size of neighbourhood (pixel neighbourhood)
    :param k: int - number of splits (crossvalidation runs to be performed)
    :return: list of lists - split_indices

    This function performs a repetitive (k times) split of test and train data of the MATRIX with along the first dimension y.
    First, the data is split into k stripes of equal size.
    Then for each of the stripes, all those pixels, which do not have a full neighborhood in the same stripe,
    are filtered (depends on the PATCHSIZE).
    The returned list contains k lists. In each entry a list contains [train_indices, test_indices].
    '''

    y, x, _, _ = matrix.shape
    cut = int((patchsize - 1) / 2)
    yr = np.arange(cut, y - cut, 1)

    split_indices = []

    for ki in range(k):
        if ki * (x / k) > 0:
            train_upper_indices = list(itertools.product(np.arange(cut, int(ki * (x / k)) - cut, 1), yr))
        else:
            train_upper_indices = []

        test_indices = list(itertools.product(np.arange(int(ki * (x / k)) + cut, int((ki + 1) * (x / k)) - cut, 1), yr))

        if (ki + 1) * (x / k) != x:
            train_lower_indices = list(itertools.product(np.arange(int((ki + 1) * (x / k) + cut), x - cut, 1), yr))
        else:
            train_lower_indices = []

        split_indices.append([np.array(train_upper_indices + train_lower_indices), np.array(test_indices)])

    return split_indices


def filter_indices(indices, labelmatrix, allowed_labels=[1, 2, 3, 4, 5], train=False, k=20000):
    '''

    :param indices: list - [(x1,y1), (x2,y2), ...]: indices which identify pixel (pixels or a pixel) in the labelmatrix
    :param labelmatrix: numpy.array - a matrix with labels (see function get_labels())
    :param allowed_labels: list - labels which are of interest
    :param train: boolean - whether or not subsampling should be done returning an equal amount of indices per class (only relevant for training)
    :param k: int - number of samples per class that should be returned if train=True
    :return: tuple - (filtered indices, filtered labels)

    This function filters a list of INDICES according to the label that each index is associated with in (within) the LABELMATRIX.
    If this label is not in the list of ALLOWED_LABELS, it is filtered.
    If TRAIN is false, then the filtered list and the associated labels are returned as two lists.
    Else, both are randomly filtered with respect to k samples per class and a uniformly distributed array of indices and labels is returned.
    '''


    filtered_indices = []
    labels = []
    count = 0
    for i in indices:
        y, x = i
        if labelmatrix[x, y] in allowed_labels:
            filtered_indices.append([x, y])
            labels.append(labelmatrix[x, y])
        else:
            count += 1

    if train:
        set_c = np.array(list(set(labels)))
        t_indices = []
        t_labels = []
        for c in set_c:
            current = np.array(filtered_indices)[np.array(labels) == c]
            random.shuffle(current)
            t_indices += list(current[:k])
            t_labels += [c] * k
        return np.array(t_indices), np.array(t_labels)

    return filtered_indices, labels




