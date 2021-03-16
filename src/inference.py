#import numpy as np
#from PIL import Image
#import matplotlib.pyplot as plt
import tensorflow as tf
from helpers import Transformer, evaluate
import pickle 


"""
# retrieve vocabulary
dir_Flickr_text = "./Flickr_Data/train_captions.txt"
file = open(dir_Flickr_text, 'r')
text = file.read()
train_captions = text.split('\n')
file.close()

# load image features model
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
image_features_extract_model.save('./Flickr_Data/image_features_extract_model.h5')

top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                 oov_token="<unk>",
                                                 filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# save tokenizer
with open('./Flickr_Data/tokenizer.pickle', 'wb') as handle:
   pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

# load image features model
image_features_extract_model = tf.keras.models.load_model('./Flickr_Data/image_features_extract_model.h5')

# load tokenizer 
with open('./Flickr_Data/tokenizer.pickle', 'rb') as file:
   tokenizer = pickle.load(file)

# create transformer model
top_k = 5000
num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
row_size = 8
col_size = 8
target_vocab_size = top_k + 1
dropout_rate = 0.1

transformer = Transformer(
   num_layer,
   d_model,
   num_heads,
   dff,
   row_size,
   col_size,
   target_vocab_size, 
   max_pos_encoding=target_vocab_size,
   rate=dropout_rate
)

transformer.load_weights('./weights/image_caption_transformer_weights')

# caption image
image = 'https://tensorflow.org/images/surf.jpg'
image_extension = image[-4:]
image_path = tf.keras.utils.get_file('image' + image_extension,
                                     origin=image)

caption, result, attention_weights = evaluate(image_path, 
                                              tokenizer, 
                                              image_features_extract_model,
                                              transformer)

print('Predicted Caption:', ' '.join(caption))
#temp_image = np.array(Image.open(image_path))
#plt.imshow(temp_image)