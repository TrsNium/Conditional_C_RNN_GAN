
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
        self.p_g_loss = tf.reduce_mean(tf.squared_difference(self.p_g_logits, self.real_x)) + self.p_r_loss
        
        #training models
        self.gen = Generator(args, self.z_inputs, self.l_inputs, name="Generator", reuse = True)
        self.fake_x, self.g_state, _ = self.gen._logits()

        self.real_dis = Discriminator(args, self.real_x, self.l_inputs, name="Discriminator", reuse=False)
        self.fake_dis = Discriminator(args, self.fake_x, self.l_inputs, name="Discriminator", reuse=True)
        
        d_r, self.d_r_state = self.real_dis._logits()
        d_f, self.d_f_state = self.fake_dis._logits()
        
        d_r = tf.reshape(d_r, [-1, 1])
        d_f = tf.reshape(d_f, [-1, 1])

        self.d_loss = - tf.reduce_mean(tf.log(d_r)) - tf.reduce_mean(tf.log(tf.ones_like(d_f) - d_f))
        self.g_loss = tf.reduce_mean(tf.log(tf.ones_like(d_f) - d_f))

        tf.summary.scalar("pre_train_loss", self.p_g_loss)
        tf.summary.scalar("discriminator_loss", self.d_loss)
        tf.summary.scalar("generator_loss", self.g_loss)
    
    def _feed_state(self, t_state, state, feed_dict):
        for i, (c, h) in enumerate(t_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        return feed_dict

    def train(self):
        optimizer_g_p = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.p_g_loss)
        #optimizer_d_p = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.p_d_loss)

        optimizer_g = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.g_loss)
        optimizer_d = tf.train.GradientDescentOptimizer(self.args.d_lr).minimize(self.d_loss)
         
        train_func = mk_train_func(self.args.batch_size, self.args.step_num, self.args.max_time_step, self.args.fs, self.args.range)
    
        config = tf.ConfigProto(device_count = {'GPU': 1})
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            train_graph = tf.summary.FileWriter("./logs", sess.graph)
            merged_summary = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
        
            if self.args.pretraining and not self.args.pre_train_done:
                print("pre-training開始")
                saver_ = tf.train.Saver(tf.global_variables())
                
                feches = {
                    "out": self.p_g_loss,
                    "g_loss": self.p_g_loss,
                    "optimizer_g": optimizer_g_p,
                    "final_state": self.p_gen.p_g_state
                }

                for itr, (data, label) in enumerate(train_func()):
                    inputs_, labels_ = mk_pretrain_batch(self.args.step_num, self.args.input_norm)
                    g_loss_ = 0.
                    d_loss_ = 0.
                    state_ = sess.run(self.gen.state_)
                    for step in range(self.args.step_num-1):
                        feed_dict ={}
                        feed_dict = self._feed_state(self.gen_state_, state_, feed_dict)
                        feed_dict[self.real_x] = data[:,step*self.args.max_time_step:(step+1)*self.args.max_time_step,:]
                        feed_dict[self.l_inputs] = label
                        feed_dict[self.z_inputs] = np.random.rand(self.args.batch_size, self.args.max_time_step, self.args.z_dim)
                        vals = sess.run(feches, feed_dict)    
                        state_ = vals["final_state"]
                        
                        g_loss_ += vals["g_loss"]
                        out = vals["out"]
    
                        out = np.transpose(out, (0,2,1)).astype(np.int16) 
                        [piano_roll_to_pretty_midi(out[i,:,:], self.args.fs, 0).write("./generated_mid/p_midi_{}.mid".format(i)) for i in range(self.args.batch_size)] 
                    
                    if itr % 100 == 0:print("itr", itr, "     g_loss:",g_loss_/self.args.pretrain_itrs,"     d_loss:",d_loss_/self.args.pretrain_itrs)
                    if itr % 200 == 0:saver_.save(sess, os.path.join(self.args.pre_train_path, "model.ckpt"))
                    if itr == self.args.pretrain_itrs: break
                print("pre trainingおわり")
            elif self.args.pretraining and self.args.pre_train_done:
                if not os.path.exists(self.args.pre_train_path):
                    print("checkpoint がない，始めからやり直し")
                    return

                saver_ = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='Generator'))
                saver_.restore(sess, os.path.join(self.args.pre_train_path, "model.ckpt"))
                print("restoreおわり")                
            
            saver = tf.train.Saver(tf.global_variables())
            for itr, (data, label) in enumerate(train_func()):
                g_loss, d_loss = [0., 0.]
                
                g_state_ = sess.run(self.gen.state_)
                f_d_state_ = (sess.run(self.fake_dis.fw_state), sess.run(self.fake_dis.bw_state))
                r_d_state_ = (sess.run(self.real_dis.fw_state), sess.run(self.real_dis.bw_state))
                for step in range(self.args.step_num-1):
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
                    d_loss_, f_d_state_, r_d_state, _ = sess.run([self.d_loss, self.d_f_state, self.d_r_state, optimizer_d], feed_dict)
                    
                    g_loss += g_loss_
                    d_loss += d_loss_
            
                g_loss /= self.args.step_num
                d_loss /= self.args.step_num
                if itr % 5 == 0:
                    print(itr , ":   g_loss:", g_loss, "   d_loss:", d_loss)
                    #train_graph.add_summary(summary, itr)

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
            for step in range(self.args.max_time_step_num):
                feed_dict={}
                for i, (c, h) in enumerate(self.gen.state_):
                    feed_dict[c] = state_[i].c
                    feed_dict[h] = state_[i].h
             
                atribute = np.array([[a+[random.random() for _ in range(self.args.random_dim)] for _ in range(self.args.max_time_step)] for a in [self.args.atribute_inputs]*self.args.batch_size])
                feed_dict[self.atribute_inputs] = atribute
                print(atribute)
                fake_, state_ = sess.run([self.fake, self.gen.final_state], feed_dict)
                results.append(fake_)

            results = np.transpose(np.concatenate(results, axis=1), (0,2,1)).astype(np.int16) 
            print(results.shape)
            print(np.max(results , axis=-1))
            [piano_roll_to_pretty_midi(results[i,:,:]*127, self.args.fs, 0).write("./generated_mid/midi_{}.mid".format(i)) for i in range(self.args.batch_size)]    
            print("Done check out ./generated_mid/*.mid" )
            return np.transpose(results, (0,2,1))
