from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
import tensorflow as tf

# PARAMETERS FOR MODEL #
#
#  embedding_dim  : number of features in embedding matrix
#  vocab_size     : number of words in the vocabulary 


# Word2Vec model 
#
# Model trains parameters is two seperate embeddings :
# 
#    target_embedding  : matrix of size (vocab size x number of features in embedding),
#                        maps target word to vector space
#    context_embedding : matrix of size (vocab szie x number of features in embedding),
#                        maps context word to vector space 
#
# Inputs :
#
#    vocab_size     : number of words in the vocabulary
#    embedding_dim  : number of features in the embedding 
#
# Methods:
#    __init__
#    call           : returns vector of scores for each (target,context) pair
#
class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim, num_ns):
    ''' 
       PARAMETERS SHAPES 
       target_embedding is shape (batch_size x 1 word x embedding_dim)
       context_embedding is shape (batch_size x embedding_dim x [number neg samples + 1])

       After embedding matrices created, take dot product along embedding dimensions
    '''
    super(Word2Vec, self).__init__()  #<-Extends model class 
    self.target_embedding = Embedding(vocab_size, 
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding", )
    self.context_embedding = Embedding(vocab_size, 
                                       embedding_dim, 
                                       input_length=num_ns+1)
    self.dots = Dot(axes=(3,2))
    self.flatten = Flatten()

  def call(self, pair):
    '''
           Create embedding matrices  
        -> Take dot product between the features of the 
           target word and the features of each context word 
        -> Flatten array of scores (which are just non-normalized cosine similarities)
           between each (target, context) pair to be fed to a scoring function
    '''
    target, context = pair
    we = self.target_embedding(target)
    ce = self.context_embedding(context)
    dots = self.dots([ce, we])
    return self.flatten(dots)

# ~~~~~~~~~~~~~~ # 
# LOSS FUNCTION  #
# ~~~~~~~~~~~~~~ #

# Bernoulli RV loss function, 
# using sigmoid Activation function 
def sigmoid_loss(y_true,x_logit):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=tf.cast(y_true,x_logit.dtype))