import string, re
import tensorflow as tf 
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# PARAMETERS 
# 
#  window_size        : number of words before (and after) target word to 
#                        call context words. If window_size = 1, then 
#                        create (target, context) pairs for (t-1,t+1) words
#  num_ns             : number of negative samples for each (target,context) pair 
#  vocab_size         : size of vocabulary
#  sequence_length    : number of words in each line of our text sent
#  remove_stop_words  : if True, then remove stop words 
#  seed               : random seed 


# Implements down sampling procedure of high frequency words
#  adapted from Mikolov 2013, where we generate probability
#  of keeping a target word in the training data 
#
# Function is built to be passed to tf skipgrams function 
#  using the sampling_table method 
def create_sampling_table(text_vector_ds 
                        , t_param=10**(-5)):
    '''
    Inputs :
        text_vector_ds : tf tensor of strings
        t_param        : threshold parameter determine how much to down-weight high frequency words 
    
    Outputs : 
        sample_prob : dictionary mapping word index to probability (numeric between 0 and 1)
                        of being included in the sample
    '''
    # calculate word frequencies
    word_freq=calc_word_freqs(text_vector_ds)
    # get total number of words in corpus
    corpus_size=0
    for index, freq in word_freq.items():
        corpus_size+=freq
    # get probability for each word in dictionary
    sample_prob=[None]*(max(word_freq.keys())+1)
    for index, freq in word_freq.items():
        sample_prob[index]=(((freq/corpus_size)/t_param)**(1/2)+1)*(t_param/(freq/corpus_size))
    sample_prob[0]=0
    return(sample_prob)

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed, sampling_table):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Iterate over all sequences (sentences) in dataset.
  for sequence in sequences:
        
    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence.numpy(), 
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples 
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1, 
          num_sampled=num_ns, 
          unique=True, 
          range_max=vocab_size, 
          seed=seed, 
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

# Custom standardization function to lowercase the text and 
# remove punctuation.
def custom_standardization(input_data,
                           remove_stop_words=False):
  # Make strings lower case 
  lowercase = tf.strings.lower(input_data)
  # Remove punctuation from strings
  lowercase = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')
  if remove_stop_words:
      # Remove stop words 
      from nltk.corpus import stopwords 
      eng_stopwords = stopwords.words('english')
      for stopword in eng_stopwords:
        lowercase = tf.strings.regex_replace(lowercase,
                                            '%s' % stopword, '')
  return(lowercase)

def get_text_vectorize_layer(standization_function,vocab_size,sequence_length):
    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Set output_sequence_length length to pad all samples to same length.
    vectorize_layer = TextVectorization(
        standardize=standization_function,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)
    return(vectorize_layer)

def calc_word_freqs(text_vector_ds):
    from collections import Counter
    word_freq=Counter()
    for elem in text_vector_ds:
        word_freq.update(elem.numpy())
    return(word_freq)