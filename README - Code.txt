The project was coded using Python 3.5. and contains four .py sheets. 
Both Python 3.5. as well as the used packages must be installed for the code to work.
We suggest to download the Anaconda Framework which contains both Python 3.5. as well as most relevant libraries and multiple IDEs:

Anaconda - https://www.continuum.io/Downloads

All files from this folder must be put into the python working directory(*see comment) and then from the python shell (ipython - if anaconda is installed) the test run will be executed using:

"%run full_test.py"

Regarding the functionality of the code, each sheet contains both comments as well as docstrings that explain each function. 
Further all required input files are also to be found in this folder:

-labels: ['city_inter.png', field_inter.png', 'forest_inter.png', 'grassland_inter.png', 'street_inter.png']

-data: oph_lexi.rat (NOTE: THIS FILE IS MISSING DUE TO ITS SIZE - IT COULD NOT BE SENT WITH MAIL; IT MUST BE ALSO PUT INTO THE PYTHON WORKING DIRECTORY FOR THE CODE TO WORK PROPERLY)

Example output is also contained in the folder which is a visual representation of the crossvalidated prediction results. 

The code is structured as follows:

a) RandomFerns.py 
This sheet contains 
- the FernEnsemble class 
-the functions for feature computation and  
-threshold computation using gini

b)Preprocessing.py
This sheet contains:
- a function (get_labels()) to load the labels from the five images (city_inter.png, field_inter.png, ...)
- a function (test_train_split()) which enables crossvalidation splitting the data into multiple chunks and repeatedly grouping the chunks as test- and traindata.
- a function (filter_indices()) which filters a list indices according to the labels of interest. This is relevant since some pixels (--> indices) are not labelled. These must be filtered and should be ignored both for training as for testing. Further, this function subsamples the input indices equally according to each class if train=True is specified in input. 

c)DataLoader.py
This sheet contains only a single function which loads the original rat file and returns the data in three possible formats (scatter vector, covariance matrix, coherency matrix)

d)full_test.py
In this sheet, all functions are used for a crossvalidation run of Random Ferns on coherency matrix with the parameters patchsize=5; fernnumber = 80; fernsize = 4;
Further, presumming is implemented here using a patch of size (4,2), effectively reducing the number of pixels in the image. For the label of the (4,2) region of presuming, the mode label is used. 


