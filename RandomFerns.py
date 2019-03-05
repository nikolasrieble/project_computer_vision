import itertools
import random
from operator import add
import numpy as np


def combine(a):
    '''
    This function returns an array of size a[0] which results from elementwise multiplication
    of all arrays contained in a
    :param a: list of distributions over classes in an array - one distribution per fern
    :return: final prediction - elementwise multiplied
    '''

    res = a[0]
    i = 1
    while True:
        try:
            res = res * a[i]
            i += 1
        except:
            return res



def logeuclid(m, n):
    '''import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


data_wf, labels_wf = get_data()

#im = plt.imshow(data, interpolation='none')

im = plt.imshow(labels_wf, cmap='gist_ncar', aspect='auto')

values = list(set(np.array(labels_wf).flatten()))
# get the colors of the values, according to the
# colormap used by imshow
colors = [ im.cmap(im.norm(value)) for value in values]
# create a patch (proxy artist) for every color

explanation = ['not labelled',
               'city',
                'field',
                'forest',
                'grassland',
                'street']

patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=explanation[i]) ) for i in range(len(values)) ]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

plt.show()
    Compute the logeuclid distance between two arrays of 3*3 matrices elementwise
    :param m: numpy.array of matrices A
    :param n: numpy.array of matrices B
    :return: numpy.array of logeuclid distances (is it? or it is L1-norm?)
    '''
    X = np.log(m / n)
    dist = np.sqrt(np.trace(np.einsum('aij, ajk->aki', X.transpose(0, 2, 1), X), axis1=1, axis2=2))
    return dist.real



def srwd(m, n):
    '''
    Computes the symetric (symmetric) revised wishart distance between two arrays of complex-valued 3*3 matrices
    This function is hardcoded and highly optimized for 3*3 matrices.
    :param m: numpy.array of matrices 1
    :param n: numpy.array of matrices 2
    :return: numpy.array of symetric (symmetric) revised wishart distance
    '''
    m = np.moveaxis(np.reshape(m, m.shape[:-2] + (-1,)), -1, 0)
    n = np.moveaxis(np.reshape(n, n.shape[:-2] + (-1,)), -1, 0)
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = m
    n1, n2, n3, n4, n5, n6, n7, n8, n9 = n
    return 0.5 * np.einsum("i...,i...->...",
                           m / (m1 * (m5 * m9 - m6 * m8) + m4 * (m3 * m8 - m2 * m9) + m7 * (m2 * m6 - m3 * m5))
                           + n / (n1 * (n5 * n9 - n6 * n8) + n4 * (n3 * n8 - n2 * n9) + n7 * (n2 * n6 - n3 * n5)),
                           [m5 * n9 - m6 * n8, m6 * n7 - m4 * n9, m4 * n8 - m5 * n7, n3 * m8 - n2 * m9,
                            n1 * m9 - n3 * m7,
                            n2 * m7 - n1 * m8, m2 * n6 - m3 * n5, m3 * n4 - m1 * n6, m1 * n5 - m2 * n4]).real - 3


def bartlett(A, B):
    '''
    Computes the bartlett distance between two numpy.arrays of complex-valued 3*3 matrices
    This function is hardcoded and highly optimized for 3*3 matrices.
    :param m: numpy.array of matrices 1
    :param n: numpy.array of matrices 2
    :return: numpy.array of bartlett distance
    '''
    """computes the bartlett distance for two 3*3 matrices elementwise"""
    A = np.moveaxis(np.reshape(A, A.shape[:-2] + (-1,)), -1, 0)
    B = np.moveaxis(np.reshape(B, B.shape[:-2] + (-1,)), -1, 0)
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = A
    b1, b2, b3, b4, b5, b6, b7, b8, b9 = B
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = (A + B)

    return np.log((m1 * m5 * m9 + m4 * m8 * m3 + m7 * m2 * m6 - m1 * m6 * m8 - m3 * m5 * m7 - m2 * m4 * m9) ** 2
                  / (((a1 * a5 * a9 + a4 * a8 * a3 + a7 * a2 * a6 - a1 * a6 * a8 - a3 * a5 * a7 - a2 * a4 * a9)
                      * (
                      b1 * b5 * b9 + b4 * b8 * b3 + b7 * b2 * b6 - b1 * b6 * b8 - b3 * b5 * b7 - b2 * b4 * b9)) * 2 ** 6)).real


def euclid_dist_rgb(A, B):
    '''
    Computes the L1_Norm between two (n,1) numpy.arrays of (3,1) arrays
    :param A: array of arrays
    :param B: array of arrays
    :return:
    '''
    return np.sum(np.abs(A - B), axis=1)


def best_split(feature_values, labels):
    '''
    This function computes the best possible threshold according to the gini coefficient (it is gini impurity and not gini coefficient) as an impurity measure.
    :param feature_values: list of feature values (n,1)
    :param labels: list of labels (n,1)
    :return: float - threshold
    '''
    # training for each node/feature determining the threshold
    feature_values, labels = np.array(feature_values), np.array(labels)

    impurity = []
    possible_thresholds = np.unique(feature_values)

    num_labels = labels.size

    # the only relevant possibilities for a threshold are the feature values themselves except (except for) the lowest value

    for threshold in possible_thresholds:
        # split node content(content of node split) based on threshold
        # to do here: what happens if len(right) or len(left) is zero
        selection = feature_values <= threshold

        right = labels[selection]
        left = labels[~selection]

        num_right = right.size
        num_left = num_labels - num_right

        # compute distribution of labels for each split
        _, right_distribution = np.unique(right, return_counts=True)
        _, left_distribution = np.unique(left, return_counts=True)

        # compute impurity of split based on the distribution
        gini_right = 1 - np.sum((np.array(right_distribution) / num_right) ** 2)
        gini_left = 1 - np.sum((np.array(left_distribution) / num_left) ** 2)

        # compute weighted total impurity of the split
        gini_split = (num_right * gini_right + num_left * gini_left) / num_labels

        impurity.append(gini_split)

    # returns the threshold with the highest associated impurity value --> best split threshold
    return possible_thresholds[np.argmin(impurity)]


def feature(data, indices, fp, reference, dist):
    '''
    This function computes the feature values for each target pixel in INDICES which refer to a pixel in the image DATA.
    Each feature is determined by the relative coordinates FP(feature parameters) which is(form a tuple) a tuple and(and by the distance...) the distance
    function to be used (:) DIST (DIST or dist?).
    :param data: target image
    :param indices: target pixels in image
    :param fp: feature parameter as tuple ((x1,x2),(y1,y2), ref), where x1,x2,y1,y2 are the relative coordinates according to the target pixel
    and ref is an index which is only relevant in case x1=y1 and x2=y2 --> for single cell features.
    :param reference: a list of reference matrices
    :param dist: a string of either ['srwd', 'bartlett', 'logeuclid', 'rgb']
    :return: an array of feature values which corresponds to the list of indices
    '''

    target_A = np.array(indices) + np.array(fp[0])
    A = np.array([data[x, y] for x, y in target_A])

    if fp[0] != fp[1]:
        target_B = np.array(indices) + np.array(fp[1])
        B = np.array([data[x, y] for x, y in target_B])
    else:
        if dist is 'rgb':
            B = np.zeros((len(indices), 3))
        else:
            B = np.empty((len(indices), 3, 3)) + reference[fp[2]]

    if dist is 'srwd':
        return srwd(A, B)
    elif dist is 'bartlett':
        return bartlett(A, B)
    elif dist is 'logeuclid':
        return logeuclid(A, B)
    elif dist is 'rgb':
        return euclid_dist_rgb(A, B)


class FernEnsemble:
    def __init__(self, patchsize, number_of_ferns, fernsize, dist):
        '''
        In (During) initialization, the features are generated based on the patchsize and each fern
        is assigned a random selection of features (and a random selection of features is assigned to each fern).
        :param patchsize: size of (of pixel neighborhood)neighborhood - determining the number of possible features
        :param number_of_ferns: number of ferns
        :param fernsize: number of features per fern
        :param dist: choice of distance measure such as in ['srwd', 'bartlett', 'logeuclid', 'rgb']
        '''

        # assuming a square patch, the patch size is the size of one side
        #  of the square in pixels - must be uneven number
        assert (patchsize % 2 == 1), 'Patchsize must be an uneven (odd) number - only square (squared) ferns are implemented'
        self.number_of_ferns = number_of_ferns
        self.fernsize = fernsize
        self.patchsize = patchsize

        '''
        here a list of all possible features - determined by the patchsize - is created:
        each feature has 4 parameters: x1,y1 determining the position of the first pixel in the patch and
        x2,y2 determining the position of the second pixel in the patch
        if x1=x2 and y1=y2, then the feature is a single pixel (cell) feature. Therefore the list contains all poss.(write the whole world) features
        '''
        cut = int((patchsize - 1) / 2)
        all_pixels = list(itertools.product(list(range(-cut, cut + 1, 1)), repeat=2))
        two_cell_feats = list(itertools.combinations(all_pixels, r=2))
        single_cell_feats = [(i, i, idx) for idx, i in enumerate(all_pixels)]

        self.all_poss_features = single_cell_feats + two_cell_feats
        '''
        shuffeling (shuffling) the list of possible features introduces randomness in selection
        after shuffling simply popping the first element is a random choice
        '''
        random.shuffle(self.all_poss_features)
        '''for each fern - determined by number_of_ferns - an amount of random features
        is drawn from the list all_poss_features'''

        ferns = []
        for i in range(number_of_ferns):
            fern_features = []
            for j in range(fernsize):
                # for each fern, take random fernsize features
                fern_features.append(self.all_poss_features.pop(0))
            ferns.append(fern_features)
        self.ferns = ferns

        '''initialize attributes that are to be filled in training'''
        self.classes = None
        self.leafs = None
        self.threshold = None
        self.single_cell_reference = None
        self.dist = dist

    def train(self, data, indices, labels, laplace=1, split='median'):
        '''
        For all features, the best threshold is determined and the distribution per leaf is learned based on a method of choice
        :param data: the whole image
        :param indices: a list of target indices - determining the training set
        :param labels: the labels associated with the indices
        :param laplace: the coefficient for laplace correction (maybe writing also the value of the coefficient?)
        :param split: method of choice for determination of node_split as in ['best', 'median'], where 'best' refers to gini-coefficent (gini impurity)
        :return: None (yet the attributes .leafs and .threshold are determined)
        '''

        'choose a random pixel cell matrix for the single cell features'
        if self.dist is 'rgb':
            self.single_cell_reference = [data[i[0], i[1]] for i in random.sample(list(indices), self.patchsize ** 2)]
        else:
            self.single_cell_reference = [data[i[0], i[1], :, :] for i in
                                          random.sample(list(indices), self.patchsize ** 2)]
        'initialization with uniform distribution for each leaf'
        self.classes = np.unique(labels)
        initial_distribution = [1 / self.classes.size] * self.classes.size
        self.leafs = np.array([[initial_distribution] * (2 ** self.fernsize)] * self.number_of_ferns)

        #learning procedure starts
        all_thresholds = []
        for fern_index, fern in enumerate(self.ferns):

            fern_thresholds = []
            patch_to_leaf = [str(0)] * len(indices)

            for feature_index, params in enumerate(fern):

                feature_values = feature(data, indices, params, self.single_cell_reference, self.dist)

                if split == 'best':
                    threshold = best_split(feature_values, labels)
                elif split == 'median':
                    threshold = np.median(feature_values)
                fern_thresholds.append(threshold)

                patch_to_leaf_f = (feature_values <= threshold).astype(int).astype(str)
                patch_to_leaf = list(map(add, patch_to_leaf, patch_to_leaf_f))

            all_thresholds.append(fern_thresholds)
            leaf_index = [int(i, 2) for i in patch_to_leaf]
            labels_to_leafs = [(g, [li for gi, li in zip(leaf_index, labels) if gi == g]) for g in set(leaf_index)]

            # compute leaf distribution
            for (leaf, labels_in_leaf) in labels_to_leafs:
                distribution = []
                for c in self.classes:
                    distribution.append(labels_in_leaf.count(c))

                # laplace correction
                distribution = [x if x != 0 else laplace for x in distribution]
                # normalizing to get probabilites
                distribution /= np.sum(distribution)
                # save leafwise distribution as attribute of fern class
                self.leafs[fern_index][leaf] = distribution

        self.threshold = all_thresholds

    def predict(self, data, indices, prediction='maximum'):
        '''
        This function uses the previously learned thresholds as well as (the) leaf distributions (,) to predict the probability
        of a sample (,) based on the leaf (in which ends up per fern) it ends in per fern. These (the instead of these) predictions are combined multiplicatively and returned.
        :param data: complete image
        :param indices: indices of test data
        :param prediction: mode of prediction - if maximum: only the index of the label with the highest probability is returned
        :return: array of predicted labels (if prediction = 'maximum') or matrix of predicted class distribution per sample (else)
        '''
        # assert that training has been done before
        if self.threshold is None:
            return 'No training done yet'

        # prediction
        ensemble_votes = [None] * self.number_of_ferns

        # exact same procedure as for learning - each fern assigns each patch to a leaf
        # and thus to a leafs (leaf's) distribution. Then for each leafs (leaf), the arg max is returned,
        # so that each fern has single vote for each patch
        # Finally all votes are returned
        for fern_index, fern in enumerate(self.ferns):
            patch_to_leaf = [str(0)] * len(indices)
            for feature_index, params in enumerate(fern):
                thres = self.threshold[fern_index][feature_index]

                feature_values = feature(data, indices, params, self.single_cell_reference, self.dist)

                # list of featurewise split [0,1] for each patch
                # patch_to_leaf_f = [str(int(i <= thres)) for i in feature_values]
                patch_to_leaf_f = (feature_values <= thres).astype(int).astype(str)
                # adding the binary 0,1 to the previous node split thus assigning each patch to a leaf
                patch_to_leaf = list(map(add, patch_to_leaf, patch_to_leaf_f))
            # from binary to integer name of leaf
            leaf_index = [int(i, 2) for i in patch_to_leaf]

            # return the distribution which has been learned during training
            ensemble_votes[fern_index] = self.leafs[fern_index][leaf_index]

            # print('fern' + str(fern_index) + ' prediction done')

        # combine all ferns prediction for all patches and return the index of the label with the highest probability
        if prediction == 'maximum':
            return self.classes[combine(ensemble_votes).argmax(axis=1)]
        else:
            return combine(ensemble_votes)

