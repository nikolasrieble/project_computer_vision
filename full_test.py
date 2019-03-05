#these files must all be in the same working directory to be explicitly loaded like this
#the import statements must most probably be adjusted (email to nikolas-rieble@gmx.de if any issues occur)
from DataLoader import *
from Preprocessing import *
from RandomFerns import *

from sklearn.metrics import confusion_matrix as cf
import skimage
from scipy import stats
import pickle

path = 'D:\\Dokumente\\Studium\\TU Berlin - ITM\\HTCV\\Random_Ferns\\Code_Development\\Data\\oph_lexi.rat'
d = loadData(path, out='coherence') # loading the rat file
labels_wf = get_labels() # load the labels

# averaging the coherence (coherency) matrices over a patch of 4*2
polsar_blocks = skimage.util.view_as_blocks(d, (4, 2, 3, 3))
polsar_reduced = np.empty((1660, 695, 3, 3), dtype=complex)
for ix in range(polsar_reduced.shape[0]):
    for iy in range(polsar_reduced.shape[1]):
        polsar_reduced[ix, iy] = np.mean(polsar_blocks[ix][iy][0][0].reshape(8,3,3), axis=0)

# get the mode of all labels over a patch of 4*2
labels_blocks = skimage.util.view_as_blocks(labels_wf, (4, 2))
labels_reduced = np.empty((1660, 695), dtype=int)
for ix in range(labels_blocks.shape[0]):
    for iy in range(labels_blocks.shape[1]):
        labels_reduced[ix, iy] = stats.mode(labels_blocks[ix][iy].flatten())[0]


# list of tuples which contains (contain) the parameters to be used such as (patchsize, fernnumber, fernsize)
params = (5, 80, 4)  # [(5, 80, 4)]
dist = 'srwd'

ps, fn, fs = params


# splitting the data into test and train data
tts = test_train_split(polsar_reduced, ps, k=5)
cv = []
for train_i, test_i in tts:
    # filter out all indices which are not labelled (label = 0) and subsample for training data
    train_indices, labels = filter_indices(train_i, labels_reduced, train=True)
    test_indices, y_test = filter_indices(test_i, labels_reduced)
    cv.append((train_indices, labels, test_indices, y_test))


y = []  # initialization of result list
confusion = np.zeros((5,5))  # confusion matrix initialization

for train_indices, labels, test_indices, y_test in cv:

    Rferns = FernEnsemble(ps, fn, fs, dist=dist)
    Rferns.train(polsar_reduced, train_indices, labels)
    y_pred = Rferns.predict(polsar_reduced, test_indices, prediction='maximum')
    y.append((y_pred, y_test))  # for each cv iteration save prediction y_pred and the true labels y_test
    confusion += cf(y_test, y_pred)  # for each iteration add the results to the confusion matrix

name = str(dist) + '_ps' + str(ps) + '_fn' + str(fn) + '_fs' + str(fs)
pickle.dump(y, open(name + '.p', "wb"))  # save the results as pickle file for later inspection

################################################################################################
####################### Visualization of Output ################################################
################################################################################################


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(num=None, figsize=(15, 15), dpi=40, facecolor='white', edgecolor='k')

# plot the correct labels in the first subplot
ax1 = plt.subplot(gs[0, 1])
im = ax1.imshow(labels_reduced, cmap='gist_ncar', aspect='auto')
values = list(set(np.array(labels_wf).flatten()))
colors = [im.cmap(im.norm(value)) for value in values]
explanation = ['not labelled',
               'city',
               'field',
               'forest',
               'grassland',
               'street']
patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=explanation[i])) for i in range(len(values))]
# put those patched as legend-handles into the legend
ax1.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# plot the predicted labels in the second subplot
target = np.zeros((1660, 695))
for id_cv, (train_indices, labels, test_indices, y_test) in enumerate(cv):

    for id_x, test_i in enumerate(test_indices):
        target[test_i[0], test_i[1]] = y[id_cv][0][id_x]

ax2 = plt.subplot(gs[0, 0])
ax2.imshow(target, cmap='gist_ncar', aspect='auto')

# plot the confusion matrix in the third subplot
ax3 = plt.subplot(gs[1, 0])

classes = ['city',
           'field',
           'forest',
           'grassland',
           'street']

cm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]  # normalization of confusion matrix
ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax3.set_title('Confusion matrix')

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, np.round(cm[i, j], 2),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# plot the correct-incorrect comparison in the fourth subplot
target = np.zeros((1660, 695))
for id_cv, (train_indices, labels, test_indices, y_test) in enumerate(cv):

    for id_x, test_i in enumerate(test_indices):
        if y_test[id_x] == y[id_cv][0][id_x]:
            target[test_i[0], test_i[1]] = 1
        else:
            target[test_i[0], test_i[1]] = -1

ax4 = plt.subplot(gs[1, 1])
im = ax4.imshow((np.array(target)), aspect='auto', cmap='RdYlGn')
values = np.unique(target.ravel())
colors = [im.cmap(im.norm(value)) for value in values]

patches = [mpatches.Patch(color=colors[i], label=['false', 'ignored', 'correct'][idx]) for idx, i in
           enumerate(range(len(values)))]
plt.legend(handles=patches, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)

left = 0.125  # the left side of the subplots of the figure
right = 0.9  # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9  # the top of the subplots of the figure
wspace = 0.2  # the amount of width reserved for blank space between subplots
hspace = 0.2  # the amount of height reserved for white space between subplots

plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
plt.suptitle('Patchsize: '+str(ps)+' Fernnumber: '+str(fn)+' Fernsize: '+str(fs)+' with '+str(dist)+' distance', fontsize=20)

plt.show()