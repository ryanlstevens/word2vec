# 3rd party libraries
from functools import partial
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
import sys, os, io 

# udf libraries
import model 
import processing_h

# path to file 
path_to_file=os.path.join(paths_dict['data'],'train_text.csv') if len(sys.argv)<=1 else sys.argv[1]


# parameters 
data_processing_parameters = {'BATCH_SIZE'  : 1024 if len(sys.argv)<=1 else int(sys.argv[2]),
                              'BUFFER_SIZE' : 5000 if len(sys.argv)<=1 else int(sys.argv[3]),
                              'AUTOTUNE' : tf.data.experimental.AUTOTUNE if len(sys.argv)<=1 else int(sys.argv[4])}
word2vec_parameters = {'NUM_NS' : 4 if len(sys.argv)<=1 else int(sys.argv[5]),
                       'T_PARAM' : 10**(-5) if len(sys.argv)<=1 else float(sys.argv[6]),
                        'VOCAB_SIZE' : 1000 if len(sys.argv)<=1 else int(sys.argv[7]),
                        'WINDOW_SIZE' : 5 if len(sys.argv)<=1 else int(sys.argv[8]),
                        'EMBEDDING_DIM' : 128 if len(sys.argv)<=1 else int(sys.argv[9]),
                        'SEQUENCE_LENGTH' : 10 if len(sys.argv)<=1 else int(sys.argv[10])}
optimization_parameters = {'EPOCHS' : 20 if len(sys.argv)<=1 else int(sys.argv[11]),
                           'SEED' : 142 if len(sys.argv)<=1 else int(sys.argv[12])}

model_run_values=(['{k}={v}'.format(k=k,v=v) for k,v in data_processing_parameters.items()]+
                  ['{k}={v}'.format(k=k,v=v) for k,v in word2vec_parameters.items()]+
                  ['{k}={v}'.format(k=k,v=v) for k,v in optimization_parameters.items()])
model_run_values='_'.join(model_run_values)                  
path_to_embedding_tsv=os.path.join(sys.argv[13],'embedding_{0}.tsv'.format(model_run_values))
path_to_metadata_tsv=os.path.join(sys.argv[13],'metadata_{0}.tsv'.format(model_run_values))

# Read csv data in as tf TextLineDataset
train_ds = tf.data.TextLineDataset(path_to_file)

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Set output_sequence_length length to pad all samples to same length.
custom_standardization_f=partial(processing_h.custom_standardization,remove_stop_words=True)
vectorize_layer = TextVectorization(
    standardize=custom_standardization_f,
    max_tokens=word2vec_parameters['VOCAB_SIZE'],
    output_mode='int',
    output_sequence_length=word2vec_parameters['SEQUENCE_LENGTH'])

# Create vocabulary
vectorize_layer.adapt(train_ds.batch(data_processing_parameters['BATCH_SIZE']))

### Generate vectors for each element in our text dataset

# Vectorize function requires 1-d array
@tf.autograph.experimental.do_not_convert
def vectorize_text(text):
    text = tf.expand_dims(text,-1)
    return(tf.squeeze(vectorize_layer(text)))

# Vectorize data set
# Batch process : take 1024 observations -> vectorize -> unbatch them
text_vector_ds = train_ds.batch(data_processing_parameters['BATCH_SIZE']).prefetch(data_processing_parameters['AUTOTUNE']).map(vectorize_text).unbatch()

# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Generate training data 
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Create vocabulary
targets, contexts, labels = processing_h.generate_training_data(
    sequences=text_vector_ds, 
    window_size=word2vec_parameters['WINDOW_SIZE'], 
    num_ns=word2vec_parameters['NUM_NS'], 
    vocab_size=word2vec_parameters['VOCAB_SIZE'],
    sampling_table=processing_h.create_sampling_table(text_vector_ds,t_param=word2vec_parameters['T_PARAM']),
    seed=optimization_parameters['SEED'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Configure dataset for performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Create dataset from tensors
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))

# Shuffle data + batch 
dataset = dataset.shuffle(data_processing_parameters['BUFFER_SIZE']).batch(data_processing_parameters['BATCH_SIZE'], drop_remainder=True)

# Add cache + prefetch to improve performance
# Cache deals with opening file + pointing to file
# Prefetch will fetch data on seperate thread while model trains (it produces, while model consumes)
dataset = dataset.cache().prefetch(buffer_size=data_processing_parameters['AUTOTUNE'])


# ~~~~~~~~~~~~~~~~~~~~~~ #
# Build + Train Word2Vec 
# ~~~~~~~~~~~~~~~~~~~~~~ #

# Compile model layers 
word2vec = model.Word2Vec(word2vec_parameters['VOCAB_SIZE'], word2vec_parameters['EMBEDDING_DIM'],word2vec_parameters['NUM_NS'])
word2vec.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model
word2vec.fit(dataset, epochs=optimization_parameters['EPOCHS'])

# ~~~~~~~~~~~~~~~~~~~~~ #
# Save model
# ~~~~~~~~~~~~~~~~~~~~~ # 


# Get embedding layer to save
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

# Open file object
out_v = io.open(path_to_embedding_tsv, 'w', encoding='utf-8')    #<- Embedding layer
out_m = io.open(path_to_metadata_tsv, 'w', encoding='utf-8')     #<- Vocabulary 

for index, word in enumerate(vocab):
  if  index == 0: continue # skip 0, it's padding.
  vec = weights[index] 
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()
