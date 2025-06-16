from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
from features import extract_features_omniglot_hierarchical, extract_features_mini_imagenet_hierarchical
from hierarchical_memory import HierarchicalVariationalMemory, hierarchical_variational_loss
from inference import inference_block
from utilities import *
from data import get_data
import os


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["Omniglot", "miniImageNet", 'tieredImageNet', 'cifarfs'],
                        default="miniImageNet", help="Dataset to use")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                        help="Whether to run training only, testing only, or both training and testing.")
    parser.add_argument("--seed", type=int, default=42, help="dataset seeds")
    parser.add_argument("--d_theta", type=int, default=256, help="Size of the feature extractor output.")
    parser.add_argument("--num_samples", type=int, default=10, help="Size of the random feature base.")
    parser.add_argument("--shot", type=int, default=1, help="Number of training examples.")
    parser.add_argument("--way", type=int, default=5, help="Number of classes.")
    parser.add_argument("--test_shot", type=int, default=None, help="Shot to be used at evaluation time.")
    parser.add_argument("--test_way", type=int, default=None, help="Way to be used at evaluation time.")
    parser.add_argument("--tasks_per_batch", type=int, default=4, help="Number of tasks per batch.")
    parser.add_argument("--memory_samples", type=int, default=50, help="Number of samples from memory.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top attention weights.")
    parser.add_argument("--num_classes", type=int, default=64, help="Number of training classes.")
    parser.add_argument("--test_iterations", type=int, default=10, help="test_iterations.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.00025, help="Learning rate.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations.")
    parser.add_argument("--checkpoint_dir", "-c", default='./checkpoint_hvm', help="Directory to save trained models.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout keep probability.")
    parser.add_argument("--test_model_path", "-m", default='./checkpoint_hvm/best_validation', help="Model to load and test.")
    parser.add_argument("--print_freq", type=int, default=10, help="Frequency of summary results.")
    parser.add_argument("--load_dir", "-lc", default='', help="Directory to save trained models.")
    parser.add_argument("--aug", type=bool, default=False, help="data augmentation")
    
    # HVM specific parameters
    parser.add_argument("--num_levels", type=int, default=4, help="Number of hierarchical levels")
    parser.add_argument("--kl_weight", type=float, default=0.1, help="Weight for KL divergence loss")
    parser.add_argument("--hierarchical", type=bool, default=True, help="Use hierarchical memory")
    
    args = parser.parse_args()

    # adjust test_shot and test_way if necessary
    if args.test_shot is None:
        args.test_shot = args.shot
    if args.test_way is None:
        args.test_way = args.way

    return args


def build_hierarchical_model(args, data):

    

    if args.dataset == "miniImageNet" or args.dataset == 'tieredImageNet':
        feature_extractor_fn = extract_features_mini_imagenet_hierarchical
        num_levels = 5 if args.num_levels > 5 else args.num_levels
    else:
        feature_extractor_fn = extract_features_omniglot_hierarchical
        num_levels = 4 if args.num_levels > 4 else args.num_levels
    

    hvm = HierarchicalVariationalMemory(
        num_classes=args.num_classes,
        d_theta=args.d_theta,
        num_levels=num_levels,
        memory_size=args.memory_samples,
        top_k=args.top_k
    )
    

    train_images = tf.placeholder(tf.float32, [None, None, 
                                               data.get_image_height(),
                                               data.get_image_width(),
                                               data.get_image_channels()],
                                  name='train_images')
    test_images = tf.placeholder(tf.float32, [None, None,
                                              data.get_image_height(),
                                              data.get_image_width(),
                                              data.get_image_channels()],
                                 name='test_images')
    train_labels = tf.placeholder(tf.float32, [None, None, args.way], name='train_labels')
    test_labels = tf.placeholder(tf.float32, [None, None, args.way], name='test_labels')
    dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
    
    def hierarchical_few_shot_learning(inputs):

        train_inputs, test_inputs, train_outputs, test_outputs = inputs
        

        with tf.variable_scope('shared_features', reuse=tf.AUTO_REUSE):
            train_features_hierarchical = feature_extractor_fn(
                images=train_inputs,
                output_size=args.d_theta,
                use_batch_norm=True,
                dropout_keep_prob=dropout_keep_prob
            )
            test_features_hierarchical = feature_extractor_fn(
                images=test_inputs,
                output_size=args.d_theta,
                use_batch_norm=True,
                dropout_keep_prob=dropout_keep_prob
            )
        

        prototypes = {}
        memory_params = {}
        class_logits = {}
        kl_losses = []
        

        prev_latent_memory = None
        prev_prototype = None
        
        for level in range(1, num_levels + 1):
            level_name = f'level_{level}'
            
            if level_name not in train_features_hierarchical:
                continue
                

            support_features = train_features_hierarchical[level_name]
            query_features = test_features_hierarchical[level_name]
            

            support_prototypes_per_class = []
            latent_memories_per_class = []
            
            for c in range(args.way):

                class_mask = tf.equal(tf.argmax(train_outputs, 1), c)
                class_features = tf.boolean_mask(support_features, class_mask)
                

                latent_memory_mu, latent_memory_logvar = hvm.hierarchical_memory_recall(
                    support_features=class_features,
                    support_labels=None,
                    level=level,
                    prev_latent_memory=prev_latent_memory
                )
                

                prototype_mu, prototype_logvar = hvm.hierarchical_prototype_inference(
                    support_features=class_features,
                    latent_memory=latent_memory_mu,
                    prev_prototype=prev_prototype,
                    level=level
                )
                
                support_prototypes_per_class.append(prototype_mu)
                latent_memories_per_class.append(latent_memory_mu)
                

                if level_name not in prototypes:
                    prototypes[level_name] = {'mu': [], 'logvar': []}
                    memory_params[level_name] = {'mu': [], 'logvar': []}
                
                prototypes[level_name]['mu'].append(prototype_mu)
                prototypes[level_name]['logvar'].append(prototype_logvar)
                memory_params[level_name]['mu'].append(latent_memory_mu)
                memory_params[level_name]['logvar'].append(latent_memory_logvar)
                

                kl_prototype = -0.5 * tf.reduce_sum(1 + prototype_logvar - tf.square(prototype_mu) - tf.exp(prototype_logvar))
                kl_memory = -0.5 * tf.reduce_sum(1 + latent_memory_logvar - tf.square(latent_memory_mu) - tf.exp(latent_memory_logvar))
                kl_losses.append(kl_prototype + kl_memory)
            

            level_prototypes = tf.concat(support_prototypes_per_class, axis=0)
            

            query_expanded = tf.expand_dims(query_features, axis=1)
            prototype_expanded = tf.expand_dims(level_prototypes, axis=0)
            distances = tf.reduce_sum(tf.square(query_expanded - prototype_expanded), axis=-1)
            class_logits[level_name] = -distances
            

            if latent_memories_per_class:
                prev_latent_memory = tf.reduce_mean(tf.concat(latent_memories_per_class, axis=0), axis=0, keepdims=True)
            if support_prototypes_per_class:
                prev_prototype = tf.reduce_mean(tf.concat(support_prototypes_per_class, axis=0), axis=0, keepdims=True)
        

        weights = {}
        num_valid_levels = len(class_logits)
        uniform_weight = 1.0 / num_valid_levels if num_valid_levels > 0 else 1.0
        
        for level_name in class_logits:
            weights[level_name] = tf.constant(uniform_weight)
        

        final_logits = None
        for level_name in class_logits:
            if level_name in weights:
                weighted_logits = class_logits[level_name] * weights[level_name]
                if final_logits is None:
                    final_logits = weighted_logits
                else:
                    final_logits += weighted_logits
        
        if final_logits is None:
            final_logits = tf.zeros([tf.shape(test_inputs)[0], args.way])
        

        total_kl_loss = tf.reduce_sum(kl_losses) if kl_losses else tf.constant(0.0)
        
        return final_logits, total_kl_loss, prototypes, memory_params
    

    logits, kl_loss, prototypes, memory_params = hierarchical_few_shot_learning([
        train_images, test_images, train_labels, test_labels
    ])
    

    classification_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=test_labels)
    )
    

    total_loss = classification_loss + args.kl_weight * kl_loss
    

    predictions = tf.argmax(logits, axis=1)
    correct_predictions = tf.equal(predictions, tf.argmax(test_labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return {
        'total_loss': total_loss,
        'classification_loss': classification_loss,
        'kl_loss': kl_loss,
        'accuracy': accuracy,
        'logits': logits,
        'placeholders': {
            'train_images': train_images,
            'test_images': test_images,
            'train_labels': train_labels,
            'test_labels': test_labels,
            'dropout_keep_prob': dropout_keep_prob
        },
        'memory_placeholders': hvm.memory_keys  
    }


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    args = parse_command_line()
    

    logfile, checkpoint_path_validation, checkpoint_path_final = get_log_files(
        args.checkpoint_dir, args.mode, args.shot
    )
    

    data = get_data(args.dataset, seed=args.seed)
    

    model = build_hierarchical_model(args, data)
    

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(model['total_loss'])
    

    memory_data = {}
    for level in range(1, args.num_levels + 1):
        level_name = f'level_{level}'
        memory_data[f'memory_keys_{level_name}'] = np.random.normal(0, 1, (args.num_classes, args.d_theta))
        memory_data[f'memory_values_mean_{level_name}'] = np.random.normal(0, 1, (args.num_classes, args.d_theta))
        memory_data[f'memory_values_var_{level_name}'] = np.ones((args.num_classes, args.d_theta)) * 0.1
    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        if args.mode in ['train', 'train_test']:
            print_and_log(logfile, "start training...")
            
            for iteration in range(args.iterations):

                train_batch = data.get_batch('train', args.tasks_per_batch, args.shot, args.way, args.shot)
                

                feed_dict = {
                    model['placeholders']['train_images']: train_batch['train_images'],
                    model['placeholders']['test_images']: train_batch['test_images'],
                    model['placeholders']['train_labels']: train_batch['train_labels'],
                    model['placeholders']['test_labels']: train_batch['test_labels'],
                    model['placeholders']['dropout_keep_prob']: args.dropout
                }
                

                for key, value in memory_data.items():
                    if key in model['memory_placeholders']:
                        feed_dict[model['memory_placeholders'][key]] = value
                

                _, loss, cls_loss, kl_loss, acc = sess.run([
                    train_op,
                    model['total_loss'],
                    model['classification_loss'],
                    model['kl_loss'],
                    model['accuracy']
                ], feed_dict=feed_dict)
                
                if iteration % args.print_freq == 0:
                    message = f"迭代 {iteration}: 总损失={loss:.4f}, 分类损失={cls_loss:.4f}, KL损失={kl_loss:.4f}, 准确率={acc:.4f}"
                    print_and_log(logfile, message)
        
        if args.mode in ['test', 'train_test']:
            print_and_log(logfile, "start testing...")
            
            test_accuracies = []
            for test_iter in range(args.test_iterations):
                test_batch = data.get_batch('test', 1, args.test_shot, args.test_way, 15)
                
                feed_dict = {
                    model['placeholders']['train_images']: test_batch['train_images'],
                    model['placeholders']['test_images']: test_batch['test_images'],
                    model['placeholders']['train_labels']: test_batch['train_labels'],
                    model['placeholders']['test_labels']: test_batch['test_labels'],
                    model['placeholders']['dropout_keep_prob']: 1.0
                }
                

                for key, value in memory_data.items():
                    if key in model['memory_placeholders']:
                        feed_dict[model['memory_placeholders'][key]] = value
                
                acc = sess.run(model['accuracy'], feed_dict=feed_dict)
                test_accuracies.append(acc)
            
            mean_acc = np.mean(test_accuracies)
            std_acc = np.std(test_accuracies)
            message = f"test accuracy: {mean_acc:.4f} ± {std_acc:.4f}"
            print_and_log(logfile, message)


if __name__ == "__main__":
    tf.app.run() 