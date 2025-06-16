import tensorflow as tf
import numpy as np
from utilities import sample_normal, dense_layer, dot_product_attention
from inference import inference_block


class HierarchicalVariationalMemory:

    
    def __init__(self, num_classes, d_theta, num_levels, memory_size=50, top_k=10):

        self.num_classes = num_classes
        self.d_theta = d_theta
        self.num_levels = num_levels
        self.memory_size = memory_size
        self.top_k = top_k
        
        self.memory_keys = {}
        self.memory_values_mean = {}
        self.memory_values_var = {}
        
        for level in range(1, num_levels + 1):
            level_name = f'level_{level}'
            self.memory_keys[level_name] = tf.placeholder(tf.float32, 
                                                         [num_classes, d_theta],
                                                         name=f'memory_keys_{level_name}')
            self.memory_values_mean[level_name] = tf.placeholder(tf.float32,
                                                               [num_classes, d_theta],
                                                               name=f'memory_values_mean_{level_name}')
            self.memory_values_var[level_name] = tf.placeholder(tf.float32,
                                                              [num_classes, d_theta],
                                                              name=f'memory_values_var_{level_name}')
    
    def hierarchical_memory_recall(self, support_features, support_labels, level, prev_latent_memory=None):

        level_name = f'level_{level}'
        

        support_nu = tf.reduce_mean(support_features, axis=0, keepdims=True)
        

        memory_keys = self.memory_keys[level_name]
        memory_values_mean = self.memory_values_mean[level_name]
        memory_values_var = self.memory_values_var[level_name]
        

        dotp = tf.matmul(support_nu, memory_keys, transpose_b=True)
        attention_weights = tf.nn.softmax(dotp)
        

        top_attention_weights = tf.nn.top_k(attention_weights, self.top_k).values
        top_attention_weights = tf.nn.softmax(top_attention_weights)
        top_attention_indices = tf.nn.top_k(attention_weights, self.top_k).indices
        

        list_indices = [[top_attention_indices[0][i]] for i in range(self.top_k)]
        memory_mean_selected = tf.gather_nd(memory_values_mean, list_indices)
        memory_var_selected = tf.gather_nd(memory_values_var, list_indices)
        

        memory_mean = tf.matmul(top_attention_weights, memory_mean_selected)
        memory_var = tf.matmul(top_attention_weights, memory_var_selected)
        

        memory_samples = sample_normal(memory_mean, memory_var, self.memory_size)
        memory_samples = tf.reshape(memory_samples, (self.memory_size, self.d_theta))
        

        if prev_latent_memory is not None:

            support_nu_tile = tf.tile(support_nu, (self.memory_size, 1))
            prev_memory_tile = tf.tile(prev_latent_memory, (self.memory_size, 1))
            combined_input = tf.concat([support_nu_tile, memory_samples, prev_memory_tile], axis=-1)
        else:

            support_nu_tile = tf.tile(support_nu, (self.memory_size, 1))
            combined_input = tf.concat([support_nu_tile, memory_samples], axis=-1)
        

        with tf.variable_scope(f'latent_memory_{level_name}', reuse=tf.AUTO_REUSE):
            latent_memory_mu = inference_block(combined_input, self.d_theta, self.d_theta, 'memory_mu')
            latent_memory_logvar = inference_block(combined_input, self.d_theta, self.d_theta, 'memory_logvar')
        

        latent_memory_mu_avg = tf.reduce_mean(latent_memory_mu, axis=0, keepdims=True)
        latent_memory_logvar_avg = tf.reduce_mean(latent_memory_logvar, axis=0, keepdims=True)
        
        return latent_memory_mu_avg, latent_memory_logvar_avg
    
    def hierarchical_prototype_inference(self, support_features, latent_memory, prev_prototype=None, level=1):

        level_name = f'level_{level}'
        

        support_nu = tf.reduce_mean(support_features, axis=0, keepdims=True)
        

        if prev_prototype is not None:
            inference_input = tf.concat([support_nu, latent_memory, prev_prototype], axis=-1)
        else:
            inference_input = tf.concat([support_nu, latent_memory], axis=-1)
        

        with tf.variable_scope(f'prototype_{level_name}', reuse=tf.AUTO_REUSE):
            prototype_mu = inference_block(inference_input, self.d_theta, self.d_theta, 'prototype_mu')
            prototype_logvar = inference_block(inference_input, self.d_theta, self.d_theta, 'prototype_logvar')
        
        return prototype_mu, prototype_logvar
    
    def compute_adaptive_weights(self, support_gradients):

        weights = {}
        
        for level in range(1, self.num_levels + 1):
            level_name = f'level_{level}'
            grad = support_gradients[level_name]
            

            with tf.variable_scope(f'weight_network_{level_name}', reuse=tf.AUTO_REUSE):
                weight_logit = dense_layer(grad, 1, None, True, 'weight_logit')
                weights[level_name] = weight_logit
        

        weight_values = [weights[f'level_{i}'] for i in range(1, self.num_levels + 1)]
        weight_stack = tf.concat(weight_values, axis=-1)
        weight_normalized = tf.nn.softmax(weight_stack)
        

        for i, level in enumerate(range(1, self.num_levels + 1)):
            level_name = f'level_{level}'
            weights[level_name] = tf.expand_dims(weight_normalized[:, i], axis=-1)
        
        return weights
    
    def hierarchical_prototype_classification(self, query_features, prototypes, weights):

        logits_per_level = {}
        

        for level in range(1, self.num_levels + 1):
            level_name = f'level_{level}'
            
            if level_name in query_features and level_name in prototypes:
                query_feat = query_features[level_name]
                prototype = prototypes[level_name]
                

                distances = tf.reduce_sum(tf.square(tf.expand_dims(query_feat, axis=1) - 
                                                  tf.expand_dims(prototype, axis=0)), axis=-1)
                logits_per_level[level_name] = -distances
        

        final_logits = None
        for level in range(1, self.num_levels + 1):
            level_name = f'level_{level}'
            if level_name in logits_per_level and level_name in weights:
                level_logits = logits_per_level[level_name]
                level_weight = weights[level_name]
                
                weighted_logits = level_logits * level_weight
                
                if final_logits is None:
                    final_logits = weighted_logits
                else:
                    final_logits += weighted_logits
        
        return final_logits


def hierarchical_variational_loss(prototypes, targets, memory_params, kl_weight=1.0):

    total_loss = 0.0
    

    classification_loss = tf.constant(0.0)  
    

    kl_loss_prototypes = 0.0
    for level_name in prototypes:
        if 'mu' in prototypes[level_name] and 'logvar' in prototypes[level_name]:
            mu = prototypes[level_name]['mu']
            logvar = prototypes[level_name]['logvar']
            
            
            kl_div = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
            kl_loss_prototypes += kl_div
    
    
    kl_loss_memory = 0.0
    for level_name in memory_params:
        if 'mu' in memory_params[level_name] and 'logvar' in memory_params[level_name]:
            mu = memory_params[level_name]['mu']
            logvar = memory_params[level_name]['logvar']
            
            
            kl_div = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
            kl_loss_memory += kl_div
    
    total_loss = classification_loss + kl_weight * (kl_loss_prototypes + kl_loss_memory)
    
    return total_loss, kl_loss_prototypes, kl_loss_memory 