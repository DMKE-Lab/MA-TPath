import numpy as np
import tensorflow as tf
class Agent(object):
    def __init__(self, params):
        self.relation_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.tim_vocab_size = len(params['tim_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = tf.constant(params['entity_vocab']['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)
        self.tPAD = tf.constant(params['tim_vocab']['PAD'], dtype=tf.int32)
        if params['use_entity_embeddings']:
            self.entity_initializer = tf.contrib.layers.xavier_initializer()
        else:
            self.entity_initializer = tf.zeros_initializer()
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.train_tims = params['train_tim_embeddings']
        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_relation = tf.constant(np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])
        self.dummy_start_tim = tf.constant(np.ones(self.batch_size, dtype='int64') * params['tim_vocab']['DUMMY_START_TIM'])
        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2
        with tf.variable_scope("relation_lookup_table"):
            self.relation_embedding_placeholder = tf.placeholder(tf.float32, [self.relation_vocab_size, 2 * self.embedding_size])
            self.relation_lookup_table = tf.get_variable("relation_lookup_table",
                                                         shape=[self.relation_vocab_size, 2 * self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer(),
                                                         trainable=self.train_relations)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.relation_embedding_placeholder)
        with tf.variable_scope("tim_lookup_table"):
            self.tim_embedding_placeholder = tf.placeholder(tf.float32, [self.tim_vocab_size, 2 * self.embedding_size])
            self.tim_lookup_table = tf.get_variable("tim_lookup_table",
                                                        shape=[self.tim_vocab_size, 2 * self.embedding_size],
                                                        dtype=tf.float32,
                                                        initializer=tf.contrib.layers.xavier_initializer(),
                                                        trainable=self.train_tims)
            self.tim_embedding_init = self.tim_lookup_table.assign(self.tim_embedding_placeholder)
        with tf.variable_scope("entity_lookup_table"):
            self.entity_embedding_placeholder = tf.placeholder(tf.float32, [self.entity_vocab_size, 2 * self.embedding_size])
            self.entity_lookup_table = tf.get_variable("entity_lookup_table",
                                                       shape=[self.entity_vocab_size, 2 * self.entity_embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)
        with tf.variable_scope("policy_step"):
            cells = []
            for _ in range(self.LSTM_Layers):
                cells.append(tf.contrib.rnn.LSTMCell(self.m * self.hidden_size, use_peepholes=True, state_is_tuple=True))
            self.policy_step = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            self.policy_step_tim = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)
    def policy_MLP(self, state):
        with tf.variable_scope("MLP_for_policy"):
            hidden = tf.layers.dense(state, 4 * self.hidden_size, activation=tf.nn.relu)
            output = tf.layers.dense(hidden, self.m * self.embedding_size, activation=tf.nn.relu)
        return output
    def policy_MLPs(self, state):
        with tf.variable_scope("MLPs_for_policy"):
            hidden = tf.layers.dense(state, 4 * self.hidden_size, activation=tf.nn.relu)
            output = tf.layers.dense(hidden, self.m * self.embedding_size, activation=tf.nn.relu)
        return output
    def policy_MLP_ENT(self, state):
        with tf.variable_scope("MLP_for_policy_ent"):
            hidden = tf.layers.dense(state, 4 * self.hidden_size, activation=tf.nn.relu)
            output = tf.layers.dense(hidden,2 * self.embedding_size, activation=tf.nn.relu)
        return output
    def action_encoder(self, next_relations, next_entities):
        with tf.variable_scope("lookup_table_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding
    def relation_encoder(self,next_relations):
        with tf.variable_scope("lookup_table_rel_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)  #[B,D]
        return relation_embedding
    def entity_encoder(self,next_entities):
        with tf.variable_scope("lookup_table_ent_edge_encoder"):
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)  #[B,D]
        return entity_embedding
    def tim_encoder(self, next_tims, next_entities):
        with tf.variable_scope("lookup_table_edge_encoder"):
            tim_embedding = tf.nn.embedding_lookup(self.tim_lookup_table, next_tims)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([tim_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = tim_embedding
        return action_embedding
    def step(self, next_relations, next_tims, next_entities, prev_state, prev_relation, prev_tim, query_embedding, query_embedding_tim, current_entities,
             label_action, label_tim, range_arr, first_step_of_test):
        prev_relation_embedding = self.action_encoder(prev_relation, current_entities)
        prev_tim_embedding = self.tim_encoder(prev_tim, current_entities)
        output, new_state = self.policy_step(prev_relation_embedding, prev_state)
        output_tim, new_state_tim = self.policy_step_tim(prev_tim_embedding, prev_state)
        prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table, current_entities)
        if self.use_entity_embeddings:
            state = tf.concat([output, prev_entity], axis=-1)
            state_tim = tf.concat([output_tim, prev_entity], axis=-1)
        else:
            state = output
            state_tim = output_tim
        candidate_relation_embeddings = self.action_encoder(next_relations, next_entities)
        candidate_tim_embeddings = self.tim_encoder(next_tims, next_entities)
        state_query_concat = tf.concat([state, query_embedding], axis=-1)
        state_query_concat_tim = tf.concat([state_tim, query_embedding_tim], axis=-1)
        state_ent = tf.concat([output, prev_entity], axis=-1)
        state_ent_query_concat = tf.concat([state_ent, query_embedding], axis=-1)
        output_ent = self.policy_MLP_ENT(state_ent_query_concat)
        output_expanded_ent = tf.expand_dims(output_ent, axis=1)
        output_rel = self.policy_MLP(state_query_concat)
        output_expanded = tf.expand_dims(output_rel, axis=1)
        # output = self.policy_MLP(state_query_concat)
        
        output_expanded = tf.expand_dims(output, axis=1)
        prelim_scores = tf.reduce_sum(tf.multiply(candidate_relation_embeddings, output_expanded), axis=2)
        prelim_scores_tim = tf.reduce_sum(tf.multiply(candidate_tim_embeddings, output_expanded), axis=2)
        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD  # matrix to compare
        mask = tf.equal(next_relations, comparison_tensor)  # The mask
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = tf.where(mask, dummy_scores, prelim_scores)  # [B, MAX_NUM_ACTIONS]
        comparison_tensor_tim = tf.ones_like(next_tims, dtype=tf.int32)* self.tPAD
        mask_tim = tf.equal(next_tims, comparison_tensor_tim)
        dummy_scores_tim = tf.ones_like(prelim_scores) * -99999.0
        scores_tim = tf.where(mask_tim, dummy_scores_tim, prelim_scores_tim)
        action = tf.to_int32(tf.multinomial(logits=scores, num_samples=1))  # [B, 1]
        tim = tf.to_int32(tf.multinomial(logits=scores_tim, num_samples=1))
        label_tim = tf.squeeze(tim, axis=1)
        label_action = tf.squeeze(action, axis=1)
        loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)  # [B,]
        loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_tim, labels=label_tim)
        loss = tf.add(loss1, loss2)
        action_idx = tf.squeeze(action)
        tim_idx = tf.squeeze(tim)
        chosen_relation = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))
        chosen_tim = tf.gather_nd(next_tims, tf.transpose(tf.stack([range_arr, tim_idx])))
        temp = tf.ones([1,next_relations.shape[1]],dtype=tf.int32)
        chosen_relation_expand = tf.multiply(action, temp)
        re = tf.bitwise.bitwise_xor(next_relations, chosen_relation_expand)
        re1 = tf.equal(re, 0)
        temp = tf.ones_like(next_entities, dtype=tf.int32)
        next_filter_entities = tf.where(re1,next_entities,temp)
        candidate_entity_embeddings = self.entity_encoder(next_filter_entities)
        prelim_ent_scores = tf.reduce_sum(tf.multiply(candidate_entity_embeddings, output_expanded_ent), axis=2)
        comparison_tensor = tf.ones_like(next_filter_entities, dtype=tf.int32) * self.rPAD
        mask = tf.equal(next_filter_entities, comparison_tensor)
        dummy_scores = tf.ones_like(prelim_ent_scores) * -99999.0
        ent_scores = tf.where(mask, dummy_scores, prelim_ent_scores)
        ent_action = tf.to_int32(tf.multinomial(logits=ent_scores, num_samples=1))
        ent_label_action =  tf.squeeze(ent_action, axis=1)
        ent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ent_scores, labels=ent_label_action)
        chosen_triple = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))

        return loss, new_state, tf.nn.log_softmax(scores+scores_tim), action_idx, tim_idx, chosen_relation, chosen_tim

    def __call__(self, candidate_relation_sequence, candidate_tim_sequence, candidate_entity_sequence, current_entities,
                 path_label, path_label_tim, query_relation, query_tim, range_arr, first_step_of_test, T=3, entity_sequence=0):
        self.baseline_inputs = []
        self.candidate_tim_sequence_i = candidate_tim_sequence
        query_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, query_relation)
        query_embedding_tim = tf.nn.embedding_lookup(self.tim_lookup_table, query_tim)
        state = self.policy_step.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        prev_relation = self.dummy_start_relation
        prev_tim = self.dummy_start_tim
        all_loss = []
        all_logits = []
        tim_idx = []
        all_rel_loss = []
        all_ent_loss = []
        all_rel_logits = []
        all_ent_logits = []
        action_idx = []
        with tf.variable_scope("policy_steps_unroll") as scope:
            for t in range(T):
                if t > 0:
                    scope.reuse_variables()
                next_possible_relations = candidate_relation_sequence[t]
                next_possible_entities = candidate_entity_sequence[t]
                next_possible_tims = candidate_tim_sequence[t]
                current_entities_t = current_entities[t]
                path_label_t = path_label[t]
                path_label_ti = path_label_tim[t]
                loss, state, logits, act_idx, ti_idx, chosen_relation, chosen_tim = self.step(next_possible_relations, next_possible_tims,
                                                                              next_possible_entities,
                                                                              state, prev_relation, prev_tim, query_embedding,query_embedding_tim,
                                                                              current_entities_t,
                                                                              label_action=path_label_t,
                                                                              label_tim=path_label_ti,
                                                                              range_arr=range_arr,
                                                                              first_step_of_test=first_step_of_test)
                all_rel_loss.append(loss)
                all_ent_loss.append(loss)
                all_rel_logits.append(logits)
                all_ent_logits.append(logits)
                all_loss.append(loss)
                all_logits.append(logits)
                action_idx.append(act_idx)
                tim_idx.append(ti_idx)
                prev_relation = chosen_relation
                prev_tim = chosen_tim
        return all_loss, all_logits, action_idx, tim_idx
