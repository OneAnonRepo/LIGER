'''
Created on Nov 25, 2018

@author:
'''

import json
import os
import random
import time
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

from utility import MLP, ThreadedIterator

class HybridDiscriminativeModel(object):   
    
    @staticmethod
    def default_params():
        return {
            'num_labels': 35,
            
            'num_epochs': 3000,
            'patience': 25,
            'learning_rate': 0.001,
            'clamp_gradient_norm': 1.0,
            
            'rnn_state_dropout_keep_prob': 1,
            'mlp_dropout_keep_prob': 1,

            'vocabulary_embedding_size': 100,
            'hidden_size': 100,
 
            'random_seed': 0,
            
            'number_of_programs': 140,
            'number_of_paths': 1,
            'number_of_executions': 1,
                        
            'variable_gap': 4, 
            'variable_limit': 8, 
            'state_gap': 50, 
            'state_limit': 200,
            
            'test_variable_gap': 5, 
            'test_variable_limit': 5, 
            'test_state_gap': 50, 
            'test_state_limit': 50,
            
            
            'states_encoding_rnn_cell': 'GRU',  # GRU, CudnnCompatibleGRUCell, or RNN
            'states_encoding_rnn_activation': 'tanh',  # tanh, ReLU
            
            'states_reduction_rnn_cell': 'GRU',  # GRU, CudnnCompatibleGRUCell, or RNN
            'states_reduction_rnn_activation': 'tanh',  # tanh, ReLU
            
            'execution_to_program' : "reduce_max", #reduce_max, reduce_mean, reduce_sum
            
            
            'data_dir': /Data/Hybrid',
            'log_dir': /Log/Hybrid',
            'model_dir': /Model/Hybrid',

            'train_file': 'train.json',
            'valid_file': 'valid.json',
            'test_file': 'test.json',
            
            'dict_file': 'dict.json',
            
            'model_file': 'model.ckpt',                    
        }
    
    def __init__(self):
        # Collect parameters:
        params = HybridDiscriminativeModel.default_params()
               
        self.params = params
        print("Starting with following parameters:\n%s" % (json.dumps(self.params)))
        
        self.best_model_file = os.path.join(self.params["model_dir"], self.params['model_file'])
        # load dictionary 
        dict_path = os.path.join(self.params["data_dir"], params['dict_file'])            
        with open(dict_path, 'r') as f:
            dict_words = json.load(f)
        self.vocabulary = dict_words

        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])
        
        self.train_data_db = self.load_data(self.params['variable_gap'], self.params['variable_limit'], 
                                            self.params['state_gap'], self.params['state_limit'], 
                                            self.params['data_dir'], self.params['train_file'])
        self.valid_data_db = self.load_data(self.params['test_variable_gap'], self.params['test_variable_limit'], 
                                            self.params['test_state_gap'], self.params['test_state_limit'],
                                            self.params['data_dir'], self.params['valid_file'])
        self.test_data_db = self.load_data(self.params['test_variable_gap'], self.params['test_variable_limit'], 
                                           self.params['test_state_gap'], self.params['test_state_limit'], 
                                           self.params['data_dir'], self.params['test_file'])
        
        
#         self.__examine_db__(self.train_data_db)
#         self.__examine_db__(self.valid_data_db)
#         self.__examine_db__(self.test_data_db)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            
            self.placeholders = {}

            self.ops = {}

            self.make_model()
            self.make_train_step()
 
            self.sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()  


    def __examine_db__(self, db):
        for var_key in db.keys():
            for state_key in db[var_key]:            
                print ("variable number %i and state number %i has %i programs" %(var_key, state_key, len(db[var_key][state_key]['programs'])))
                print ("The max variable number is %i" % (db[var_key][state_key]['max_var']))
                print ("The max token number is %i" % (db[var_key][state_key]['max_token']))
                print ("The max state_statement number is %i" % (db[var_key][state_key]['max_statement_state']))
                print (os.linesep)        
                

    def __create_database__(self, variable_gap, variable_limit, state_gap, state_limit):
        db = {}
                 
        # programs of variable number n (n <= variable limit) will be stored in db under the key (i) 
        # that is equal to or immediate larger than n 
        for i in range(variable_gap, variable_limit+1, variable_gap):            
            db[i] = {}      
             
            # programs of state number n (n <= state_limit) will be stored in db[i] under the key (j) 
            # that is equal to or immediate larger than n 
            for j in range(state_gap, state_limit+1, state_gap):                
                db[i][j] = {'programs':[], 'max_var':0, 'max_token':0, 'max_statement_state':0}
                 
            # programs of state number n (n > state_limit) will be stored in db[i] under the last key 
            # (equals to state_limit+1)
            db[i][state_limit+1] = {'programs':[], 'max_var':0, 'max_token':0, 'max_statement_state':0}
         
        # programs of variable number n (n > variable_limit) will be stored in db under the last key 
        # (equals to variable_limit+1) 
        db[variable_limit+1] = {}
        for j in range(state_gap, state_limit+1, state_gap):        
            db[variable_limit+1][j] = {'programs':[], 'max_var':0, 'max_token':0, 'max_statement_state':0}
        db[variable_limit+1][state_limit+1] = {'programs':[], 'max_var':0, 'max_token':0, 'max_statement_state':0}
         
        return db
             
    def __find_keys_in_db__(self, db, current_var_num, current_state_number):    
        key_var = 0
        key_state = 0
         
        variable_limit = max(db.keys())
        state_limit = max(db[next(iter(db))].keys())
         
        if(current_var_num >= variable_limit):
            key_var = variable_limit
        else:
            ordered_keys = list(db.keys())
            ordered_keys.sort()
            for i in ordered_keys:
                if(i >= current_var_num):
                    key_var = i
                    break
         
        if(current_state_number >= state_limit):
            key_state = state_limit
        else:
            ordered_keys = list(db[next(iter(db))].keys())
            ordered_keys.sort()
            for i in ordered_keys:
                if(i >= current_state_number):
                    key_state = i
                    break
     
        return (key_var, key_state)
 
    def load_data(self, variable_gap, variable_limit, state_gap, state_limit, data_dir, file_name):
        db = self.__create_database__(variable_gap, variable_limit, state_gap, state_limit)
                              
        full_path = os.path.join(data_dir, file_name)    
         
        with open(full_path, 'r') as f:
            data = json.load(f)
         
        for p in data:
            p_var_num = p["number_of_vars"]
            p_token_num = p["number_of_tokens"]
            p_state_statement_num = p["number_of_states_statements"]
             
            var_key, state_key = self.__find_keys_in_db__(db, p_var_num, p_state_statement_num)
     
            batch_programs = db[var_key][state_key]
            batch_programs['programs'].append(p)
            batch_programs['max_var']=max(p_var_num,batch_programs['max_var'])
            batch_programs['max_token']=max(p_token_num,batch_programs['max_token'])            
            batch_programs['max_statement_state']=max(p_state_statement_num,batch_programs['max_statement_state'])       
                 
        return db


    def make_model(self):             
        # number of programs * number of paths * number of executions * number of states/number of statements, number of variables  
        self.placeholders["executions"] = tf.placeholder(tf.int32, [None, None], name="executions")
        # number of programs * number of paths * number of executions * number of states/number of statements
        self.placeholders["variable_number_sequence"] = tf.placeholder(tf.int32, [None], name="variable_number_sequence")

        # number of programs * number of paths * number of statements/number of states, number of tokens           
        self.placeholders["tokens"] = tf.placeholder(tf.int32, [None, None], name="tokens")
        # number of programs * number of paths * number of statements/number of states       
        self.placeholders["tokens_number_sequence"] = tf.placeholder(tf.int32, [None], name="tokens_number_sequence")
 
        # number of programs * number of paths  
        self.placeholders["state_statement_number_sequence"] = tf.placeholder(tf.int32, [None], name="state_statement_number_sequence")
         
        self.placeholders['rnn_state_dropout_keep_prob'] = tf.placeholder(tf.float32, None, 
                                                                          name='rnn_state_dropout_keep_prob') 
 
        self.placeholders['mlp_dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='mlp_dropout_keep_prob')           
         
        self.placeholders["max_state_statement"] = tf.placeholder(tf.int32, None, name="max_state")
         
        self.placeholders['label'] = tf.placeholder(tf.int32, [None], name='label')
                                
                                
        batch_size = self.params["number_of_programs"] * self.params["number_of_paths"]
                 
                                   
        embedding_matrix = tf.get_variable('embedding_matrix', [len(self.vocabulary)+1, self.params["vocabulary_embedding_size"]]) 
        # number of programs * number of paths * number of executions * number of states/number of statements,  number of variables , embedding_size
        embedded_executions = tf.nn.embedding_lookup(params=embedding_matrix, ids=self.placeholders["executions"])
        # number of programs * number of paths * number of states/number of statements,  number of tokens , embedding_size
        embedded_tokens = tf.nn.embedding_lookup(params=embedding_matrix, ids=self.placeholders["tokens"])
        
        
        with tf.variable_scope("state_encoding"):
            state_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.params["hidden_size"]) 
            state_encoder_cell = tf.nn.rnn_cell.DropoutWrapper(state_encoder_cell, 
                                                               state_keep_prob=self.params["rnn_state_dropout_keep_prob"])
            _, states_embedding = tf.nn.dynamic_rnn(state_encoder_cell, embedded_executions, 
                                                    sequence_length=self.placeholders["variable_number_sequence"],
                                                    initial_state=state_encoder_cell.zero_state(tf.shape(embedded_executions)[0], 
                                                                                                tf.float32),
                                                    dtype=tf.float32)
        # number of programs * number of paths * number of executions, number of states/number of statements, embedding_size
        dynamic_state_embedding = tf.convert_to_tensor(tf.split(states_embedding[1], batch_size * self.params["number_of_executions"], axis=0))
        
        
        with tf.variable_scope("statement_encoding"):
            statement_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.params["hidden_size"])
            statement_encoder_cell = tf.nn.rnn_cell.DropoutWrapper(statement_encoder_cell, 
                                                                   state_keep_prob=self.params["rnn_state_dropout_keep_prob"])
            _, statements_embedding = tf.nn.dynamic_rnn(statement_encoder_cell, embedded_tokens, 
                                                        sequence_length=self.placeholders["tokens_number_sequence"],
                                                        initial_state=statement_encoder_cell.zero_state(tf.shape(embedded_tokens)[0],
                                                                                                        tf.float32),
                                                        dtype=tf.float32)
        # number of programs * number of paths, number of states/number of statements, embedding_size
        static_tokens_embedding = tf.convert_to_tensor(tf.split(statements_embedding[1], batch_size, axis=0))
        
        
        trace_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.params["hidden_size"])
        trace_encoder_cell = tf.nn.rnn_cell.DropoutWrapper(trace_encoder_cell,
                                                           state_keep_prob=self.params["rnn_state_dropout_keep_prob"])        
        trace_encoder_initial_states = trace_encoder_cell.zero_state(batch_size, tf.float32)
        
        # axis zero stands for two LSTM states: c and h 
        trace_encoder_final_states = tf.zeros([2, batch_size, self.params["hidden_size"]])
                
        loop_counter_inital = tf.constant(0)
        
        monitor_rnn_states = tf.zeros([0, batch_size, self.params["hidden_size"]])
        monitor_output = tf.zeros([0, batch_size, self.params["hidden_size"]])
        monitor_mask = tf.zeros([0, batch_size], tf.float32)
        monitor_attention_probabilities = tf.zeros([0, self.params["number_of_executions"]+1, 1], tf.float32)
 
         
        def while_condition(loop_counter,state_statement_number_sequence, rnn_states, trace_encoder_final_states, 
                            monitor_rnn_states, monitor_mask, monitor_output, monitor_attention_probabilities):
            return loop_counter < self.placeholders["max_state_statement"]
        
        def while_body(loop_counter, state_statement_number_sequence, rnn_states, trace_encoder_final_states, 
                       monitor_rnn_states, monitor_mask, monitor_output, monitor_attention_probabilities):
            loop_counter_current = loop_counter
                        
            # number of programs * number of paths * number of executions, embedding_size
            current_states = tf.gather_nd(dynamic_state_embedding, 
                                          tf.stack([tf.range(0, batch_size * self.params["number_of_executions"]), 
                                                    tf.zeros([batch_size * self.params["number_of_executions"]], tf.int32)+loop_counter_current], axis=1))   
            # number of programs * number of paths, number of executions, embedding_size
            current_states = tf.convert_to_tensor(tf.split(current_states, batch_size, axis=0))
                        
            # number of programs * number of paths, embedding_size
            current_tokens = tf.gather_nd(static_tokens_embedding, 
                                          tf.stack([tf.range(0, batch_size), tf.zeros([batch_size], tf.int32)+loop_counter_current], axis=1))
            # number of programs * number of paths, 1, embedding_size            
            current_tokens = tf.expand_dims(current_tokens, axis=1)
            
            # number of programs * number of paths, number of executions+1, embedding_size            
            current_states_and_tokens = tf.concat([current_states,current_tokens], axis=1)
            
            # number of programs * number of paths, 2 * lstm hidden_size
            rnn_states_concat = tf.concat((rnn_states[0], rnn_states[1]), axis=1)     
            # number of programs * number of paths, 1, 2 * lstm hidden_size            
            rnn_states_concat = tf.expand_dims(rnn_states_concat, axis=1)
            # 1 * number of executions+1 * 1
            replicate_factor = tf.ones([1,self.params["number_of_executions"]+1,1], tf.float32)
            # number of programs * number of paths, number of executions+1, 2 * lstm hidden_size            
            rnn_states_concat = rnn_states_concat * replicate_factor
            
            # number of programs * number of paths, number of executions+1, embedding_size + 2 * lstm hidden_size          
            rnn_inputs_and_states = tf.concat([current_states_and_tokens,rnn_states_concat], axis=-1)
            # number of programs * number of paths * number of executions+1, embedding_size + 2 * lstm hidden_size                      
            rnn_inputs_and_states = tf.concat(tf.unstack(rnn_inputs_and_states, num=batch_size, axis=0), 0)
 
            # number of programs * number of paths * number of executions+1, 1                       
            attention_scores_fn = MLP(rnn_inputs_and_states, 0, 1, self.placeholders['mlp_dropout_keep_prob'])      
            # number of programs * number of paths, number of executions+1, 1                                             
            attention_scores = tf.convert_to_tensor(tf.split(attention_scores_fn(), batch_size, axis=0))
            attention_probabilities = tf.nn.softmax(attention_scores, dim=1)
            
            monitor_attention_probabilities = tf.concat([monitor_attention_probabilities, attention_probabilities], axis=0)
            
            # number of programs * number of paths, embedding_size     
            inputs_after_attention = tf.reduce_sum(attention_probabilities * current_states_and_tokens, axis=1) 
                               
            _, rnn_states = trace_encoder_cell(inputs_after_attention, rnn_states)    
                
            monitor_rnn_states = tf.concat([monitor_rnn_states, rnn_states], axis=0)    
            
            monitor_output = tf.concat([monitor_output, tf.expand_dims(rnn_states[1], axis=0)], axis=0)
                    
            loop_counter_current += 1
            
            mask = tf.zeros([0], tf.float32)
            
            it_state_length = tf.unstack(state_statement_number_sequence, batch_size, axis=0)
            
            for each_state_length in it_state_length:
                def f1():          
                    return tf.zeros([1], tf.float32)
                def f2():                                                                                                
                    return tf.ones([1], tf.float32)
                
                result = tf.cond(tf.equal(each_state_length,loop_counter_current), f2, f1)    
                mask = tf.concat([mask, result], axis=0)
        
                
            monitor_mask = tf.concat([monitor_mask, tf.expand_dims(mask,0)], axis=0)        
                
            mask = tf.expand_dims(mask, axis=1)  
                    
            trace_encoder_final_states = trace_encoder_final_states + mask * rnn_states
        
            return [loop_counter_current, state_statement_number_sequence, rnn_states, trace_encoder_final_states, 
                    monitor_rnn_states, monitor_mask, monitor_output, monitor_attention_probabilities]
        
        
        _, _, _, self.ops['l_res'], self.ops['l_mono'], self.ops['l_mono_mask'], self.ops['l_mono_out'], self.ops['l_mono_attention'] = \
        tf.while_loop(while_condition, 
                      while_body, 
                      loop_vars=[loop_counter_inital, self.placeholders['state_statement_number_sequence'], 
                                 trace_encoder_initial_states, trace_encoder_final_states, monitor_rnn_states, 
                                 monitor_mask, monitor_output, monitor_attention_probabilities], 
                      shape_invariants=[loop_counter_inital.shape, 
                                        self.placeholders["state_statement_number_sequence"].shape, 
                                        LSTMStateTuple(tf.TensorShape([batch_size,self.params["hidden_size"]]), 
                                                       tf.TensorShape([batch_size,self.params["hidden_size"]])),
                                        trace_encoder_final_states.shape, 
                                        tf.TensorShape([None, batch_size, self.params["hidden_size"]]), 
                                        tf.TensorShape([None, batch_size]),
                                        tf.TensorShape([None, batch_size, self.params["hidden_size"]]),
                                        tf.TensorShape([None, self.params["number_of_executions"]+1, 1])])
        
        self.ops['attention'] = tf.squeeze(tf.reduce_mean(self.ops['l_mono_attention'], axis=0), axis=-1) 
        
        h_states = self.ops['l_res'][0]
#         c_states = self.ops['l_res'][1]
#         hc_conca_states = tf.concat([h_states,c_states], axis=1)
         
        state_rep = h_states
        self.ops['final_embeddings'] = tf.reduce_max(tf.stack(tf.split(state_rep, 
                                                                       self.params["number_of_programs"], axis=0), axis=0), axis=1)
         
        W_pred = tf.get_variable("weights_for_prediction", [self.params["hidden_size"], self.params["num_labels"]], tf.float32)
        b_pred = tf.get_variable("bias_for_prediction", [self.params["num_labels"]], tf.float32)
        logits = tf.matmul(self.ops['final_embeddings'], W_pred) + b_pred
        
        predictions = tf.argmax(logits, 1)
        comparisons = tf.cast(tf.equal(tf.cast(predictions,tf.int32), self.placeholders['label']),tf.float32)
        accuracy = tf.reduce_mean(comparisons)            
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.placeholders['label'], logits=logits)
         
        self.ops["predictions"] = predictions
        self.ops["comparisons"] = comparisons
        self.ops["accuracy"] = accuracy                                
        self.ops["loss"] = tf.reduce_sum(loss)        
                

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
                
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)       


    # returns the 2D numpy array for dynamic and static feature of one program
    # dynamic: number of paths * number of execution * number of states, number of variables 
    # static: number of paths * number of statements, number of tokens     
    def __padding__(self, executions_dynamic, max_var, executions_static, max_token, max_state_statement):            
        padded_executions_static = []
        padded_executions_dynamic = []
        
        number_of_variable_sequence = []

        number_of_token_sequence = []
        
        number_of_state_statement_sequence = []
                
        # each_execution_static is one executed path (consist of statements)
        for each_execution_static in executions_static:
            
            each_execution_static_list_to_numpy = []
            
            number_statements = len(each_execution_static)
            
            for each_execution_statement in each_execution_static:
                
                number_tokens = len(each_execution_statement)
                            
                padded_each_execution_statement = np.pad(each_execution_statement, (0,max_token-number_tokens), 'constant')
                each_execution_static_list_to_numpy.append(padded_each_execution_statement)
            
                number_of_token_sequence.append(number_tokens)
                
            each_execution_static_list_to_numpy = np.stack(each_execution_static_list_to_numpy,axis=0)            
            each_execution_static_list_to_numpy = np.pad(each_execution_static_list_to_numpy, ((0,max_state_statement-number_statements),(0,0)), 'constant')
            
            # the actual number to be padded (in this case number_tokens) is not important  
            # because it will get thrown away in the state/statement encoding stage
            number_of_token_sequence.extend([number_tokens] * (max_state_statement-number_statements))
            
            padded_executions_static.append(each_execution_static_list_to_numpy)
                
            number_of_state_statement_sequence.append(number_statements)                    
            
        # each_execution_dynamic is one execution (consist of states)
        for each_execution_dynamic in executions_dynamic:
            
            number_states = len(each_execution_dynamic)
            number_variables = len(each_execution_dynamic[0])
            
            padded_each_execution_states = np.pad(each_execution_dynamic, ((0,max_state_statement-number_states),(0,max_var-number_variables)), 'constant')
            padded_executions_dynamic.append(padded_each_execution_states)
            
            number_of_variable_sequence.extend([number_variables] * max_state_statement)
        
        
        numpy_dynamic = np.concatenate(padded_executions_dynamic, axis = 0)
        numpy_static = np.concatenate(padded_executions_static, axis = 0)
        
        return numpy_dynamic, number_of_variable_sequence, numpy_static, number_of_token_sequence, number_of_state_statement_sequence     


    def __supplmenting__(self, list_to_grow, length, indices=None):
        if(indices is None):        
            indices = []
            
            difference = length - len(list_to_grow)

            for _ in range(difference):
                index = random.randint(0,len(list_to_grow)-1)
                indices.append(index)
        
        for index in indices:
            list_to_grow.append(list_to_grow[index])
        
        return indices

    # return a 3D lists of dynamic and static features
    # dynamic: a list of executions, one execution is a list of states, a state is a list of variables
    # static: a list of paths, one path is a list of statements, a statement is a list of tokens    
    def __formating__(self, execution_dynamic_string, execution_static_string):                
        variable_token_separator = "_|_"
        state_statement_separator = "__||__"    
        execution_separator = "___|||___"
        path_separator = "____||||____"
        
        # storing a list of dynamic execution instances each of which consist of multiple program states
        list_of_formatted_executions_dynamic = []
        # storing a list of dynamic execution instances each of which consist of multiple statements        
        list_of_formatted_executions_static = []
        
        list_of_dynamic_paths = execution_dynamic_string.split(path_separator)
        list_of_static_paths = execution_static_string.split(execution_separator)
        
        assert (len(list_of_dynamic_paths) == len(list_of_static_paths)), "dynamic and static paths are not equal"
        
        if(len(list_of_dynamic_paths) < self.params["number_of_paths"]):                       
            indices = self.__supplmenting__(list_of_dynamic_paths, self.params["number_of_paths"])            
            _       = self.__supplmenting__(list_of_static_paths, self.params["number_of_paths"], indices)
        else:
            index_set = set()
            
            for _ in range(self.params["number_of_paths"]):                
                while(True):
                    index = random.randint(0,len(list_of_dynamic_paths)-1)                    
                    if(index not in index_set):
                        index_set.add(index)
                        break
            
            sublist_of_dynamic_paths = []
            sublist_of_static_paths = []
            
            for i in index_set:
                sublist_of_dynamic_paths.append(list_of_dynamic_paths[i])
                sublist_of_static_paths.append(list_of_static_paths[i])
                
            list_of_dynamic_paths, list_of_static_paths = sublist_of_dynamic_paths, sublist_of_static_paths
                
            
        for one_dynamic_path, one_static_path in zip(list_of_dynamic_paths, list_of_static_paths):
            
            list_of_dynamic_executions = one_dynamic_path.split(execution_separator)
            
            if(len(list_of_dynamic_executions) < self.params["number_of_executions"]):
                self.__supplmenting__(list_of_dynamic_executions, self.params["number_of_executions"])                
            else:                
                np.random.shuffle(list_of_dynamic_executions)
 
                list_of_dynamic_executions = list_of_dynamic_executions[:self.params["number_of_executions"]]
            
            for one_dynamic_execution in list_of_dynamic_executions:
                list_of_formatted_states = []        
                list_of_states = one_dynamic_execution.split(state_statement_separator)
                
                assert (len(list_of_states) != 0), "Execution %i has 0 states!" % list_of_dynamic_executions.index(one_dynamic_execution)
                
                for state in list_of_states:
                    list_of_formatted_variables = []
                    list_of_variables = state.split(variable_token_separator)
                    assert (len(list_of_variables) != 0), "State %i has 0 variables!" % list_of_states.index(state)
                    
                    # add one for another reserved symbol
                    for variable in list_of_variables:
                        if variable in self.vocabulary:
                            variable_index = self.vocabulary.index(variable) + 1
                        else:
                            variable_index = 0
                        list_of_formatted_variables.append(variable_index)
                        
                    list_of_formatted_states.append(list_of_formatted_variables)
                
                list_of_formatted_executions_dynamic.append(list_of_formatted_states)
            
            
            list_of_formatted_statements = []        
            list_of_statements = one_static_path.split(state_statement_separator)
                            
            for statement in list_of_statements:
                list_of_formatted_tokens = []
                list_of_tokens = statement.split(variable_token_separator)
                
                assert (len(list_of_tokens) != 0), "Statement %i has 0 tokens!" % list_of_statements.index(statement)
                
                # add one for another reserved symbol
                for token in list_of_tokens:
                    if token in self.vocabulary:
                        token_index = self.vocabulary.index(token) + 1
                    else:
                        token_index = 0
                    list_of_formatted_tokens.append(token_index)
                    
                list_of_formatted_statements.append(list_of_formatted_tokens)
            
            list_of_formatted_executions_static.append(list_of_formatted_statements)
        
        return list_of_formatted_executions_dynamic, list_of_formatted_executions_static
                           

    def make_minibatch_iterator(self, data_db, is_training):            
            
        for var_key in data_db:        
            for state_key in data_db[var_key]:
                batch_programs = data_db[var_key][state_key]
                
                programs = batch_programs['programs']
                if(len(programs) == 0):
                    continue
                                
                remainder = len(programs) % self.params["number_of_programs"]
                if(remainder > 0):
                    self.__supplmenting__(programs, len(programs)+self.params["number_of_programs"]-remainder)             
                    
                if is_training:
                    np.random.shuffle(programs)
                    
                for i in range(0, int(len(programs)/self.params["number_of_programs"])):
                    current_batch_programs = programs[i*self.params["number_of_programs"]:(i+1)*self.params["number_of_programs"]]
                                    
                    padded_dynamic_programs = []
                    number_of_variable_sequence_programs = []

                    padded_static_programs = []      
                    number_of_token_sequence_programs = []                                  
                    
                    number_of_state_statement_sequence_programs = []
                    
                    for p in current_batch_programs:
#                         print("Current feeding program %s into a mini-batch" % p["name"])                        
                        
                        execution_dynamic_lists, executions_static_lists = self.__formating__(p["execution"], p["tokens"])

                        padded_results = self.__padding__(execution_dynamic_lists, batch_programs['max_var'], 
                                                          executions_static_lists, batch_programs['max_token'], 
                                                          batch_programs['max_statement_state'])
                        
                        padded_dynamic_programs.append(padded_results[0])
                        number_of_variable_sequence_programs.extend(padded_results[1])
                        
                        padded_static_programs.append(padded_results[2])
                        number_of_token_sequence_programs.extend(padded_results[3])
                        
                        number_of_state_statement_sequence_programs.extend(padded_results[4])                    

                    dynamic_feature_representation = np.concatenate(padded_dynamic_programs, axis=0)
                    static_feature_representation = np.concatenate(padded_static_programs, axis=0)                    
                    
                    labels = [int(p["label"]) for p in current_batch_programs]
                                    
                    batch_feed_dict = {
                        self.placeholders['executions']: dynamic_feature_representation,
                        self.placeholders['variable_number_sequence']: number_of_variable_sequence_programs,
                        
                        self.placeholders["tokens"]: static_feature_representation,
                        self.placeholders["tokens_number_sequence"]: number_of_token_sequence_programs,

                        self.placeholders["max_state_statement"]: batch_programs['max_statement_state'], 
                                
                        self.placeholders['state_statement_number_sequence']: number_of_state_statement_sequence_programs,
                        self.placeholders['label']: labels
                    }
    
                    yield batch_feed_dict      

            
    def run_epoch(self, epoch_name, data_db, is_training):
        
        numbers, loss, comparisons = 0, 0, []
                
        start_time = time.time()
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data_db, is_training), max_queue_size=5)
                  
#         batch_iterator = self.make_minibatch_iterator(data_db, is_training)
        for step, batch_placeholders in enumerate(batch_iterator):
                        
            numbers += self.params["number_of_programs"]
            
            if is_training:
                batch_placeholders[self.placeholders['rnn_state_dropout_keep_prob']] = self.params['rnn_state_dropout_keep_prob']
                batch_placeholders[self.placeholders['mlp_dropout_keep_prob']] = self.params['mlp_dropout_keep_prob']
                fetch_list = [self.ops['loss'], self.ops['accuracy'], self.ops['comparisons'], self.ops['attention'], self.ops['train_step']]
            else:
                batch_placeholders[self.placeholders['rnn_state_dropout_keep_prob']] = 1.0
                batch_placeholders[self.placeholders['mlp_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], self.ops['accuracy'], self.ops['comparisons'], self.ops['attention']]
                
            result = self.sess.run(fetch_list, feed_dict=batch_placeholders)
            (batch_loss, batch_accuracy, batch_comparison, batch_attention_sum) = (result[0], result[1], result[2], result[3])
  
            loss += batch_loss
            comparisons.extend(batch_comparison)   
            batch_dynamic_static_attention_sum = np.add.reduceat(
                batch_attention_sum,[0,self.params["number_of_executions"]], axis=0) / np.array([self.params["number_of_executions"], 1])            

            batch_attention = batch_dynamic_static_attention_sum
            dynamic_attention, static_attention = batch_attention[0], batch_attention[1]
                        
            print("Running %s, batch %i. Batch accuracy and loss are: %.4f, %.4f. Batch dynamic and static attentions are: %.4f, %.4f." 
                  % (epoch_name, step, batch_accuracy, batch_loss, dynamic_attention, static_attention))                     
    
        accuracy = np.sum(comparisons) / numbers        
                
        instance_per_sec = numbers / (time.time() - start_time)
                                  
        return loss, accuracy, instance_per_sec
    
            
#             _res,_mono,_mono_out,_loss,_mono_attention,_final_embedding = self.sess.run((
#                 self.ops['l_res'],self.ops['l_mono'],self.ops['l_mono_out'],self.ops['loss'],self.ops['l_mono_attention'], self.ops['final_embeddings']),
#                 feed_dict=batch_placeholders
    #             feed_dict={
    #                 self.placeholders["executions"]:[[3,4,20],[4,3,2],[8,3,5],[7,9,10],[2,8,5],[6,2,7],
    #                                                  [9,8,6],[3,7,2],[3,4,20],[4,3,2],[8,3,5],[7,9,10]],
    #                 self.placeholders["variable_number_sequence"]:[3,2,3,1,2,1,2,3,1,2,3,3],
    #                 self.placeholders["tokens"]:[[4,9,7],[9,4,6],[5,9,8],[2,3,4],[2,9,1],[5,7,8]], 
    #                 self.placeholders["tokens_number_sequence"]:[2,2,1,2,3,1],
    #                 self.placeholders["max_state_statement"]:3, 
    #                 self.placeholders["state_statement_number_sequence"]:[3,2], 
    #                 self.placeholders['label']:[2,3]
    #                 }
    
#                 )
        
#             print (_res)
#             print (os.linesep)
#             print (_mono_attention)
#             print (os.linesep)        
#             print (_final_embedding)          


    def train(self):

        with self.graph.as_default():
            (best_valid_accuracy, best_valid_accuracy_epoch) = (float("-inf"), 0)
     
            for epoch in range(1, self.params['num_epochs'] + 1):
                current_time = str(datetime.datetime.now().strftime("%m-%d %H:%M:%S"))
                print("\n\n================ Epoch %i (%s) ================" %(epoch, current_time))
                 
                train_loss, train_accuracy, train_speed = self.run_epoch("epoch %i (training)" % epoch, self.train_data_db, True)                
                print("\x1b[ Train: loss: %.5f | acc: %s | instances/sec: %.2f ]\n" 
                      % (train_loss, train_accuracy, train_speed))
                  
                valid_loss, valid_accuracy, valid_speed = self.run_epoch("epoch %i (validation)" % epoch, self.valid_data_db, False)
                print("\x1b[ Valid: loss: %.5f | acc: %s | instances/sec: %.2f ]" 
                      % (valid_loss, valid_accuracy, valid_speed))
      
                if valid_accuracy > best_valid_accuracy:
                    print("  (Best epoch so far, validation accuracy increases to %.5f from %.5f.)" % (valid_accuracy, best_valid_accuracy))
                    print("  (model saved to %s)\n" % self.best_model_file)
                    self.saver.save(self.sess, self.best_model_file)
                      
                    best_valid_accuracy = valid_accuracy
                    best_valid_accuracy_epoch = epoch
                      
                    test_loss, test_accuracy, test_speed = self.run_epoch("epoch %i (test)" % epoch, self.test_data_db, False)
                    print("\x1b[ Test: loss: %.5f | acc: %s | instances/sec: %.2f ]" 
                          % (test_loss, test_accuracy, test_speed))
                                          
                elif epoch - best_valid_accuracy_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params['patience'])
                    break        
         
                                         

def main():
    model = HybridDiscriminativeModel()
    model.train()     
    
if __name__=='__main__':
    main()



    
    

















