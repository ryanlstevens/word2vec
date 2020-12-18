# word2vec
Keras implementation of word2vec, including full data processing pipeline, where impelementation closely follows [TF tutorial](https://www.tensorflow.org/tutorials/text/word2vec)

Implementation primarily for building intuition for both keras and word2vec. Includes both data processing and model estimation pipelines. 

To run the data processing, must run the __submit.py__ script. This script reads in a text file (using the __path_to_text_file__ parameter), then does a grid search over parameter grids at top of script. 

If a user wants to test a single combination of parameter values, work with the commands inside the for loop, which runs preprocessing + model estimation by calling __run.py__ for a fixed set of parameter values.

__submit.py__ allows users to set a series of parameters in word2vec runs. There are three sets of parameters ``Data Processing``, ``word2Vec``, and ``Model Tuning`` parameters: 

* ``Data Processing`` parameters deal with the amount of training used in each batch, as well as, performance in pre-fetching data for each batch as the model runs. 
* ``word2vec`` parameters deal with word2vec specific parameters such as the number of negative samples.
* ``Model Tuning`` parameters deal with more general model performance issues such as number of epochs for each model run.


### DATA PROCESSING PARAMETERS
__BATCH_SIZE__  : number of training samples  
__BUFFER_SIZE__ : size of buffer to be filled while prior batch is running  
__AUTOTUNE__    : tuning parameter for number of data elements to fetch into buffer 

### DATA PROCESSING PARAMETERS
__NUM_NS__         : number of negative samples using in Noise Contrastive Estimation procedure  
__T_PARAM__        : threshold value used to down-weight high frequency words ([see equation 5](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf))  
__VOCAB_SIZE__     : number of words in vocabulary  
__WINDOW_SIZE__    : number of words before and after target word to include in context  
__EMBEDDING_DIM__  : number of hidden layers in intermediate layer (or number of vectors in word embedding)  
__SEQUENCE_LENGTH__ : length of each sentence  

### MODEL TUNING
__EPOCHS__  : number of parameter updates   
__SEED__    : random seed    


## Folder Structure

__submit.py__ runs __run.py__ in ``src`` folder, assuming the same folder structure as is in this repository. 
