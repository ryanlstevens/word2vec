import csv, os, re 
import itertools
import numpy as np
import tensorflow as tf

# Create text file dataset #

# Directory #
paths_dict={}
paths_dict['root']=os.getcwd()
paths_dict['data']=os.path.join(paths_dict['root'],'data')
paths_dict['code']=os.path.join(paths_dict['root'],'src')
paths_dict['input_data']=os.path.join(paths_dict['data'],'input')
paths_dict['output_data']=os.path.join(paths_dict['data'],'output')
paths_dict['model_runs']=os.path.join(paths_dict['output_data'],'model_runs')

# Path to csv file where each row is a sentence
#  where each word is seperated by white space character
path_to_text_file=os.path.join(paths_dict['data'],'train_text.csv')

# Set parameters for model run

# PARAMETER GRIDS 
num_ns_grid = [15,20]
vocab_size_grid = np.arange(1000,2000,500)
window_size_grid = [5,10]
embedding_dim_grid = np.arange(200,600,200)
sequence_length_grid = [10,20]
epoch_grid = [25,100]

# LOOP THROUGH GRIDS
param_grid=list(itertools.product(list(num_ns_grid),list(vocab_size_grid),list(window_size_grid),list(embedding_dim_grid),list(sequence_length_grid),list(epoch_grid)))
for ix,elem in enumerate(param_grid):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\nRunning {0} out of {1}\n'.format(ix,len(param_grid)))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
    # PARAMETERS
    # Data processing #
    BATCH_SIZE = 1024
    BUFFER_SIZE = 5000
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # word2vec specific parameters
    NUM_NS = elem[0]
    T_PARAM = 10**(-5)
    VOCAB_SIZE = elem[1]
    WINDOW_SIZE = elem[2]
    EMBEDDING_DIM = elem[3]
    SEQUENCE_LENGTH = elem[4]
    # model tuning parameters
    EPOCHS = elem[5]
    SEED = 142
    # RUN WORD 2 VEC PYTHON SCRIPT 
    run_command='''python {py_run_script} '{path_to_text_data_file}' {batch_size} {buffer_size} {autotune} {num_ns} 
                        {t_param} {vocab_size} {window_size}
                        {embedding_dim} {sequence_length}
                        {epochs} {seed} '{path_to_model_output_dir}' '''.format(
                                                        py_run_script=os.path.join(paths_dict['code'],'run.py')
                                                        ,path_to_text_data_file=path_to_text_file
                                                        ,batch_size=BATCH_SIZE
                                                        ,buffer_size=BUFFER_SIZE
                                                        ,num_ns=NUM_NS
                                                        ,t_param=T_PARAM
                                                        ,vocab_size=VOCAB_SIZE
                                                        ,window_size=WINDOW_SIZE
                                                        ,embedding_dim=EMBEDDING_DIM
                                                        ,sequence_length=SEQUENCE_LENGTH
                                                        ,epochs=EPOCHS
                                                        ,autotune=AUTOTUNE
                                                        ,seed=SEED
                                                        ,path_to_model_output_dir=paths_dict['model_runs'])
    run_command=re.sub(' +', ' ',run_command.replace('\n',''))
    os.system(run_command)