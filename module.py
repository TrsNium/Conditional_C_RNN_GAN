import tensorflow as tf


def define_cell(rnn_size, keep_prob, name):
    cell_ = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse, name)
    if keep_prob < 1.:
        cell_ = tf.contrib.rnn.DropoutWrapper(cell_, output_keep_prob=keep_prob)
    return cell_

class Generator():
    def __init__(self, args, z_inputs=None, l_inputs, name="Generator"):
        with tf.variable_scope(name) as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=args.scale))
            cell_ = tf.contrib.rnn.MultiRNNCell([define_cell(args.gen_rnn_size, args.keep_prob, "gen_{}_lstm".format(l)) for l in range(args.num_layers_g)])
         
            self.state_ = cell_.zero_state(batch_size=args.batch_size, dtype=tf.float32)
            t_state_ = self.state_
            outputs = []
            for t_ in range(args.max_time_step):
                if t_ != 0:
                    scope.reuse_variables()
                
                input_ = tf.concat([z_inputs[:,t_,:], l_inputs], axis=-1)
                rnn_input_ = tf.layers.dense(input_, args.gen_rnn_input_size, tf.nn.relu, name="RNN_INPUT_DENSE")
                _ = tf.layers.dense(x, args.gen_rnn_input_size, tf.nn.relu, name="RNN_PRE_INPUT_DENSE")
                rnn_output_, t_state_ = cell_(rnn_input_, t_state_)
                output_ = tf.layers.dense(rnn_output_, args.vocab_size, name="RNN_OUT_DENSE")
                outputs.append(output_)
       
            self.final_state = t_state_
            self.outputs = tf.transpose(tf.stack(outputs), (1,0,2))
            scope.reuse_variables()

            ##pre training
            self.state_ = cell_.zero_state(batch_size=args.batch_size, dtype=tf.float32)
            p_state_ = self.state_
            pre_train_outputs = []
            for t_ in range(args.max_time_step):
                if t_ != 0:
                    scope.reuse_variables()

                rnn_input_ = tf.layers.dense(z_inputs[:,t_,:], args.gen_rnn_input_size, tf.nn.relu, name="RNN_PRE_INPUT_DENSE")
                rnn_output_, p_state_ = cell_(rnn_input_, p_state_)
                output_ = tf.layers.dense(rnn_output_, args.vocab_size, name="RNN_OUT_DENSE")
                pre_train_outputs.append(output_)

            self.p_state = p_state_
            self.pre_train_outputs = tf.transpose(tf.stack(pre_train_outputs), (1,0,2)) 
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.reg_loss = args.reg_constant * sum(reg_losses)
    
    def _pre_train(self, y):
        loss = tf.reduce_mean(tf.squared_difference(self.pre_train_outputs, y)) + self.reg_loss
        return loss, self.p_state, self.pre_train_outputs

    def _logits(self):
        return self.outputs, self.state_

class Discriminator(object):
    def __init__(self, args, x, label, name="Discriminator", reuse=False):
        self.name = name
        self.args = args

        with tf.variable_scope(name) as scope
            if reuse:
                scope.reuse_variables()

            rnn_input_ = []
            with tf.variable_scope("rnn_in_dense") as i_scope:
                for t_ in range(args.max_time_step):
                    if t_ != 0:
                        i_scope.reuse_variables()
                
                    input_ = tf.concat([x[:,t,:], label], axis=-1)
                    rnn_input_.append(tf.layers.dense(input_, args.dis_rnn_size, tf.nn.relu))
            rnn_input_ = tf.convert_to_tensor(rnn_input_)

            fw = tf.contrib.rnn.MultiRNNCell([define_cell(args.dis_rnn_size, args.keep_prob, "dis_f_{}_lstm".format(l)) for l in range(args.num_layers_d)], state_is_tuple=True)
            bw = tf.contrib.rnn.MultiRNNCell([define_cell(args.dis_rnn_size, args.keep_prob, "dis_b_{}_lstm".format(l)) for l in range(args.num_layers_d)], state_is_tuple=True)
            self.fw_state = fw.zero_state(batch_size=args.batch_size, dtype=tf.float32)
            self.bw_state = bw.zero_state(batch_size=args.batch_size, dtype=tf.float32)
            rnn_output, self.final_state = tf.nn.bidirectional_dynamic_rnn(fw,
                                                                bw,
                                                                rnn_input_,
                                                                initial_state_fw=self.fw_state,
                                                                initial_state_bw=self.bw_state, 
                                                                dtype=tf.float32,
                                                                swap_memory = True)

            logits = []
            with tf.variable_scope("rnn_out_dense") as o_scope:
                for t_ in range(self.args.max_time_step):
                    if not t_ != 0:
                        o_scope.reuse_variables()
                    rnn_out = tf.concat([rnn_output[0][:,t_,:], rnn_output[1][:,t_,:]], axis=-1)
                    logits.append(tf.layers.dense(rnn_out, 1, name="rnn_out_dense"))
            self.logits = tf.convert_to_tensor(logits)

    
    def _logits(self):
        return self.logits, self.final_state
