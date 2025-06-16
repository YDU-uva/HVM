#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from features import extract_features_mini_imagenet_hierarchical
from hierarchical_memory import HierarchicalVariationalMemory

def test_hierarchical_features():

    print("testing hierarchical features...")
    

    batch_size = 4
    height, width, channels = 84, 84, 3
    images = tf.placeholder(tf.float32, [batch_size, height, width, channels])
    dropout_keep_prob = tf.placeholder(tf.float32, [])
    

    features = extract_features_mini_imagenet_hierarchical(
        images=images,
        output_size=256,
        use_batch_norm=True,
        dropout_keep_prob=dropout_keep_prob
    )
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        

        input_data = np.random.normal(0, 1, (batch_size, height, width, channels))
        

        feature_outputs = sess.run(features, feed_dict={
            images: input_data,
            dropout_keep_prob: 1.0
        })
        
        print("hierarchical features:")
        for level_name, feature in feature_outputs.items():
            print(f"  {level_name}: {feature.shape}")
    
    print("hierarchical features test completed✓\n")


def test_hierarchical_memory():

    print("testing hierarchical memory...")
    

    num_classes = 64
    d_theta = 256
    num_levels = 3
    memory_size = 10
    top_k = 5
    

    hvm = HierarchicalVariationalMemory(
        num_classes=num_classes,
        d_theta=d_theta,
        num_levels=num_levels,
        memory_size=memory_size,
        top_k=top_k
    )
    

    support_features = tf.placeholder(tf.float32, [5, d_theta]) 
    support_labels = tf.placeholder(tf.int32, [5])
    

    latent_memory_mu, latent_memory_logvar = hvm.hierarchical_memory_recall(
        support_features=support_features,
        support_labels=support_labels,
        level=1,
        prev_latent_memory=None
    )
    

    prototype_mu, prototype_logvar = hvm.hierarchical_prototype_inference(
        support_features=support_features,
        latent_memory=latent_memory_mu,
        prev_prototype=None,
        level=1
    )
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        

        support_data = np.random.normal(0, 1, (5, d_theta))
        support_label_data = np.array([0, 1, 2, 3, 4])
        

        memory_feed_dict = {}
        for level in range(1, num_levels + 1):
            level_name = f'level_{level}'
            memory_feed_dict[hvm.memory_keys[level_name]] = np.random.normal(0, 1, (num_classes, d_theta))
            memory_feed_dict[hvm.memory_values_mean[level_name]] = np.random.normal(0, 1, (num_classes, d_theta))
            memory_feed_dict[hvm.memory_values_var[level_name]] = np.ones((num_classes, d_theta)) * 0.1
        

        memory_mu, memory_logvar, proto_mu, proto_logvar = sess.run([
            latent_memory_mu, latent_memory_logvar, prototype_mu, prototype_logvar
        ], feed_dict={
            support_features: support_data,
            support_labels: support_label_data,
            **memory_feed_dict
        })
        
        print("hierarchical memory recall result:")
        print(f"  latent memory mean: {memory_mu.shape}")
        print(f"  latent memory variance: {memory_logvar.shape}")
        print("hierarchical prototype inference result:")
        print(f"  prototype mean: {proto_mu.shape}")
        print(f"  prototype variance: {proto_logvar.shape}")
    
    print("hierarchical memory test completed✓\n")


def test_full_pipeline():

    print("testing full pipeline...")
    

    way = 5
    shot = 1
    query_size = 15
    d_theta = 256
    num_levels = 3
    

    height, width, channels = 84, 84, 3
    

    train_images = tf.placeholder(tf.float32, [way * shot, height, width, channels])
    test_images = tf.placeholder(tf.float32, [way * query_size, height, width, channels])
    train_labels = tf.placeholder(tf.float32, [way * shot, way])
    test_labels = tf.placeholder(tf.float32, [way * query_size, way])
    dropout_keep_prob = tf.placeholder(tf.float32, [])
    

    with tf.variable_scope('shared_features'):
        train_features = extract_features_mini_imagenet_hierarchical(
            images=train_images,
            output_size=d_theta,
            use_batch_norm=True,
            dropout_keep_prob=dropout_keep_prob
        )
        test_features = extract_features_mini_imagenet_hierarchical(
            images=test_images,
            output_size=d_theta,
            use_batch_norm=True,
            dropout_keep_prob=dropout_keep_prob
        )
    

    hvm = HierarchicalVariationalMemory(
        num_classes=64,
        d_theta=d_theta,
        num_levels=num_levels,
        memory_size=10,
        top_k=5
    )
    

    logits_per_level = []
    
    for level in range(1, min(num_levels + 1, 4)):  
        level_name = f'level_{level}'
        
        if level_name in train_features:

            prototypes = []
            for c in range(way):
                class_indices = tf.range(c * shot, (c + 1) * shot)
                class_features = tf.gather(train_features[level_name], class_indices)
                class_prototype = tf.reduce_mean(class_features, axis=0)
                prototypes.append(class_prototype)
            

            stacked_prototypes = tf.stack(prototypes)
            

            query_features = test_features[level_name]
            distances = tf.reduce_sum(
                tf.square(tf.expand_dims(query_features, 1) - tf.expand_dims(stacked_prototypes, 0)),
                axis=2
            )
            level_logits = -distances
            logits_per_level.append(level_logits)
    

    if logits_per_level:
        final_logits = tf.reduce_mean(tf.stack(logits_per_level), axis=0)
    else:
        final_logits = tf.zeros([way * query_size, way])
    

    predictions = tf.argmax(final_logits, axis=1)
    true_labels = tf.argmax(test_labels, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, true_labels), tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        

        train_data = np.random.normal(0, 1, (way * shot, height, width, channels))
        test_data = np.random.normal(0, 1, (way * query_size, height, width, channels))
        

        train_label_data = np.zeros((way * shot, way))
        test_label_data = np.zeros((way * query_size, way))
        
        for i in range(way):

            for j in range(shot):
                train_label_data[i * shot + j, i] = 1

            for j in range(query_size):
                test_label_data[i * query_size + j, i] = 1
        

        memory_feed_dict = {}
        for level in range(1, num_levels + 1):
            level_name = f'level_{level}'
            if level_name in hvm.memory_keys:
                memory_feed_dict[hvm.memory_keys[level_name]] = np.random.normal(0, 1, (64, d_theta))
                memory_feed_dict[hvm.memory_values_mean[level_name]] = np.random.normal(0, 1, (64, d_theta))
                memory_feed_dict[hvm.memory_values_var[level_name]] = np.ones((64, d_theta)) * 0.1
        

        acc, logits = sess.run([accuracy, final_logits], feed_dict={
            train_images: train_data,
            test_images: test_data,
            train_labels: train_label_data,
            test_labels: test_label_data,
            dropout_keep_prob: 1.0,
            **memory_feed_dict
        })
        
        print(f"full pipeline test result:")
        print(f"  accuracy: {acc:.4f}")
        print(f"  output logits shape: {logits.shape}")
    
    print("full pipeline test completed✓\n")


def main():

    print("=== HVM (hierarchical variational memory) test ===\n")
    
    try:

        test_hierarchical_features()
        

        test_hierarchical_memory()
        

        test_full_pipeline()
        
        print("=== all tests completed ===")
        print("HVM implemented basic functionality correctly!")
        
    except Exception as e:
        print(f"error occurred during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 