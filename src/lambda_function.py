import boto3
import pickle
import tensorflow as tf
from src.helpers import Transformer, evaluate


s3_client = boto3.client('s3')

def model_setup():
    # load image features model
    image_features_extract_model = tf.keras.models.load_model('/var/task/src/Flickr_Data/image_features_extract_model.h5')

    # load tokenizer 
    with open('/var/task/src/Flickr_Data/tokenizer.pickle', 'rb') as file:
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

    transformer = Transformer(num_layer,
                              d_model,
                              num_heads,
                              dff,
                              row_size,
                              col_size,
                              target_vocab_size, 
                              max_pos_encoding=target_vocab_size,
                              rate=dropout_rate)
    transformer.load_weights('/var/task/src/weights/image_caption_transformer_weights')

    return image_features_extract_model, tokenizer, transformer


def lambda_handler(event, context):
    # extract model config
    image_features_extract_model, tokenizer, transformer = model_setup()

    # caption image
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        image_path = f"/tmp/{key.replace('/', '')}"
        s3_client.download_file(bucket, key, image_path)
        caption, result, attention_weights = evaluate(image_path, 
                                                      tokenizer, 
                                                      image_features_extract_model,
                                                      transformer)
        print(' '.join(caption))
        return caption