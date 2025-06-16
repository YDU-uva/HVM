import tensorflow as tf
from utilities import conv2d_pool_block, conv2d_transpose_layer, dense_layer, dense_block


def extract_features_omniglot_hierarchical(images, output_size, use_batch_norm, dropout_keep_prob):

    

    h1 = conv2d_pool_block(images, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_1')
    h2 = conv2d_pool_block(h1, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_2')
    h3 = conv2d_pool_block(h2, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_3')
    h4 = conv2d_pool_block(h3, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_4')


    features = {}
    features['level_1'] = tf.contrib.layers.flatten(h1)
    features['level_2'] = tf.contrib.layers.flatten(h2)
    features['level_3'] = tf.contrib.layers.flatten(h3)
    features['level_4'] = tf.contrib.layers.flatten(h4)
    
    return features


def extract_features_mini_imagenet_hierarchical(images, output_size, use_batch_norm, dropout_keep_prob):

    

    h1 = conv2d_pool_block(images, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_1')
    h2 = conv2d_pool_block(h1, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_2')
    h3 = conv2d_pool_block(h2, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_3')
    h4 = conv2d_pool_block(h3, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_4')
    h5 = conv2d_pool_block(h4, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_5')


    features = {}
    features['level_1'] = tf.contrib.layers.flatten(h1)
    features['level_2'] = tf.contrib.layers.flatten(h2)
    features['level_3'] = tf.contrib.layers.flatten(h3)
    features['level_4'] = tf.contrib.layers.flatten(h4)
    features['level_5'] = tf.contrib.layers.flatten(h5)
    
    return features


def extract_features_omniglot(images, output_size, use_batch_norm, dropout_keep_prob):


    # 4X conv2d + pool blocks
    h = conv2d_pool_block(images, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_1')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_2')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_3')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_4')

    # flatten output
    h = tf.contrib.layers.flatten(h)

    return h


def extract_features_mini_imagenet(images, output_size, use_batch_norm, dropout_keep_prob):


    # 5X conv2d + pool blocks
    h = conv2d_pool_block(images, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_1')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_2')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_3')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_4')
    h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_5')

    # flatten output
    h = tf.contrib.layers.flatten(h)

    return h


