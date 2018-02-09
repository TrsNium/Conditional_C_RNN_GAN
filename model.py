
from module import *
import os
from util import *
import numpy as np
import warnings
import random

class model():
    def __init__(self, args):
        self.args = args

        self.real_x = tf.placeholder(tf.float32, [None, args.max_time_step, args.range], "real_inputs")
        self.z_inputs = tf.placeholder(tf.float32, [None, args.max_time_step, args.z_dim])
        self.l_inputs = tf.placeholder(tf.float32, [None, args.l_dim])

        #pre training G
        self.p_gen = Generator(args, self.z_inputs, self.l_inputs, name="Generator", reuse=False)
        self.p_g_logits, self.p_g_state, self.p_r_loss = self.p_gen._logits()
        self.p_g_loss = tf.reduce_mean(tf.squared_difference(self.p_g_logits, self.real_x)) #+ self.p_r_loss
        
        self.gen = Generator(args, self.z_inputs, self.l_inputs, name="Generator", reuse = True)
        self.fake_x, self.g_state, _ = self.gen._logits()

        self.real_dis = Discriminator(args, self.real_x, self.l_inputs, name="Discriminator", reuse=False)
        self.fake_dis = Discriminator(args, self.fake_x, self.l_inputs, name="Discriminator", reuse=True)
        
        d_r, self.d_r_state = self.real_dis._logits()
        d_f, self.d_f_state = self.fake_dis._logits()
        
        #calcurate Discriminator loss 
        d_loss = []
        for t_ in range(args.max_time_step):                
            loss = -tf.reduce_mean(d_r - d_f)
            d_loss.append(loss)
        self.d_loss = tf.reduce_sum(tf.convert_to_tensor(d_loss))

        #calcurate Generator loss
        g_loss = []
        for t_ in range(args.max_time_step):
            loss = -tf.reduce_mean(d_f)
            g_loss.append(loss)
        self.g_loss = tf.reduce_sum(tf.convert_to_tensor(g_loss))

        tf.summary.scalar("pre_train_loss", self.p_g_loss)
        tf.summary.scalar("discriminator_loss", self.d_loss)
        tf.summary.scalar("generator_loss", self.g_loss)
        
        trainable_var = tf.trainable_variables()
        self.g_var = [var for var in trainable_var if "Generator" in var.name]
        self.d_var = [var for var in trainable_var if "Discriminator" in var.name]
        self.clip = [tf.clip_by_value(var, -args.c, args.c) for var in self.d_var] 

    def _feed_state(self, t_state, state, feed_dict):
        for i, (c, h) in enumerate(t_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        return feed_dict

    def train(self):
        optimizer_g_p = tf.train.AdamOptimizer(self.args.lr).minimize(self.p_g_loss, var_list=self.g_var)

        optimizer_g = tf.train.RMSPropOptimizer(self.args.lr).minimize(self.g_loss, var_list=self.g_var)
        optimizer_d = tf.train.RMSPropOptimizer(self.args.d_lr).minimize(self.d_loss, var_list=self.d_var)
         
        train_func = mk_train_func(self.args.batch_size, self.args.step_num, self.args.max_time_step, self.args.fs, self.args.range)
    
        config = tf.ConfigProto(device_count = {'GPU': 1})
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            train_graph = tf.summary.FileWriter("./logs", sess.graph)
            merged_summary = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
        
            if self.args.pretraining and not self.args.pre_train_done:
                print("-________'start pre-training'________-")
                saver_ = tf.train.Saver(tf.global_variables())
                
                for itr, (data, label) in enumerate(train_func()):
                    g_loss_ = 0.
            
                    state_ = sess.run(self.p_gen.state_)
                    for step in range(self.args.step_num):
                        feed_dict ={}
                        feed_dict = self._feed_state(self.p_gen.state_, state_, feed_dict)
                        feed_dict[self.real_x] = data[:,step*self.args.max_time_step:(step+1)*self.args.max_time_step,:]
                        feed_dict[self.l_inputs] = label
                        feed_dict[self.z_inputs] = np.random.rand(self.args.batch_size, self.args.max_time_step, self.args.z_dim)
                        out, loss, state_, _  = sess.run([self.p_g_logits, self.p_g_loss, self.p_g_state, optimizer_g_p], feed_dict)
                        g_loss_ += loss

                        out = np.transpose(out, (0,2,1)).astype(np.int16)
                        out[out>127]=127
                        [piano_roll_to_pretty_midi(out[i,:,:], self.args.fs, 0).write("./generated_mid/p_midi_{}.mid".format(i)) for i in range(self.args.batch_size)] 
                    
                    if itr % 100 == 0:print("itr", itr, "     g_loss:",g_loss_/self.args.step_num)
                    if itr % 100 == 0:print(out[out>0.])
                    if itr % 200 == 0:saver_.save(sess, os.path.join(self.args.pre_train_path, "model.ckpt"))
                    if itr == self.args.pretrain_itrs: break
                print("pre trainingおわり")
            elif self.args.pretraining and self.args.pre_train_done:
                if not os.path.exists(self.args.pre_train_path):
                    print("Doesn't exit pre trained model check point")
                    return

                saver_ = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='Generator'))
                saver_.restore(sess, os.path.join(self.args.pre_train_path, "model.ckpt"))
                print("restore done")                
            
            saver = tf.train.Saver(tf.global_variables())
            data_gen = train_func()
            for itr in range(self.args.train_itrs):
                g_loss, d_loss = [0., 0.]
                
                # train generator
                for _ in range(self.args.ncritic):
                    data, label = data_gen.__next__()
                    g_state_ = sess.run(self.gen.state_)
                    f_d_state_ = (sess.run(self.fake_dis.fw_state), sess.run(self.fake_dis.bw_state))
                    r_d_state_ = (sess.run(self.real_dis.fw_state), sess.run(self.real_dis.bw_state))
                    for step in range(self.args.step_num):
                        feed_dict = {}
                        feed_dict = self._feed_state(self.gen.state_, g_state_, feed_dict)
                        feed_dict = self._feed_state(self.fake_dis.fw_state, f_d_state_[0], feed_dict)
                        feed_dict = self._feed_state(self.fake_dis.bw_state, f_d_state_[1], feed_dict)
                        feed_dict = self._feed_state(self.real_dis.fw_state, r_d_state_[0], feed_dict)
                        feed_dict = self._feed_state(self.real_dis.bw_state, r_d_state_[1], feed_dict)
                        feed_dict[self.real_x] =  data[:, step*self.args.max_time_step:(step+1)*self.args.max_time_step,:]
                        feed_dict[self.z_inputs] = np.random.rand(self.args.batch_size, self.args.max_time_step, self.args.z_dim)
                        feed_dict[self.l_inputs] = label
                        d_loss_, f_d_state_, r_d_state, _, _ = sess.run([self.d_loss, self.d_f_state, self.d_r_state, optimizer_d, self.clip], feed_dict)
                        d_loss += d_loss_
                
                # train discriminator
                data, label = data_gen.__next__()
                g_state_ = sess.run(self.gen.state_)
                f_d_state_ = (sess.run(self.fake_dis.fw_state), sess.run(self.fake_dis.bw_state))
                r_d_state_ = (sess.run(self.real_dis.fw_state), sess.run(self.real_dis.bw_state))
                for step in range(self.args.step_num):
                    feed_dict = {}
                    feed_dict = self._feed_state(self.gen.state_, g_state_, feed_dict)
                    feed_dict = self._feed_state(self.fake_dis.fw_state, f_d_state_[0], feed_dict)
                    feed_dict = self._feed_state(self.fake_dis.bw_state, f_d_state_[1], feed_dict)
                    feed_dict = self._feed_state(self.real_dis.fw_state, r_d_state_[0], feed_dict)
                    feed_dict = self._feed_state(self.real_dis.bw_state, r_d_state_[1], feed_dict)
                    feed_dict[self.real_x] =  data[:, step*self.args.max_time_step:(step+1)*self.args.max_time_step,:]
                    feed_dict[self.z_inputs] = np.random.rand(self.args.batch_size, self.args.max_time_step, self.args.z_dim)
                    feed_dict[self.l_inputs] = label
                    g_loss_, g_state_, _ = sess.run([self.g_loss, self.gen.final_state, optimizer_g], feed_dict)
                    g_loss += g_loss_
            
                g_loss /= (self.args.step_num)
                d_loss /= (self.args.step_num)
                if itr % 5 == 0:
                    print(itr , ":   g_loss:", g_loss, "   d_loss:", d_loss/self.args.ncritic)

                if itr % 20 == 0:
                    saver.save(sess, self.args.train_path+"model.ckpt")
                    print("-------------------saved model---------------------")
                
                if itr == self.args.train_itrs:
                    break

    def generate(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, self.args.train_path+"model.ckpt")

            results = []
            state_ = sess.run(self.gen.state_)
            for step in range(self.args.step_num):
                feed_dict={}
                feed_dict = self._feed_state(self.gen.state_, state_, feed_dict)
                feed_dict[self.z_inputs] = np.random.rand(5, self.args.max_time_step, self.args.z_dim)
                feed_dict[self.l_inputs] = [[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0]]
                fake_, state_ = sess.run([self.fake_x, self.gen.final_state], feed_dict)
                results.append(fake_)

            results = np.transpose(np.concatenate(results, axis=1), (0,2,1)).astype(np.int16)
            results[results > 127] = 127
            [piano_roll_to_pretty_midi(results[i,:,:], self.args.fs, 0).write("./generated_mid/midi_{}.mid".format(i)) for i in range(5)] 
        return np.transpose(results, (0,2,1))
