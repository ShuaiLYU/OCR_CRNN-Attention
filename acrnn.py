import os
import numpy as np
import tensorflow as tf
slim=tf.contrib.slim
from config import  *
from tensorflow.python.ops import embedding_ops
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import rnn
from tensorflow.python.framework import graph_util



class ACRNN(object):

    def __init__(self,sess,param):
        self.step = 0
        self.__session = sess
        self.__learn_rate = param["learn_rate"]
        self.__learn_rate=param["learn_rate"]
        self.__max_to_keep=param["max_to_keep"]
        self.__checkPoint_dir = param["checkPoint_dir"]
        self.__restore = param["b_restore"]
        self.__mode= param["mode"]
        self.is_training=True
        self.__batch_size = param["batch_size"]
        if  self.__mode is "savaPb" :
            self.__batch_size = 1

        ################ Building graph
        with self.__session.as_default():
            (
                self.image,
                self.train_output,
                self.target_output,
                self.train_decode_result,
                self.sample_rate,
                self.pred_decode_result,
                self.loss,
                self.train_op,
                self.init_op,
            ) = self.model()

        ###############参数初始化，或者读入参数
        with self.__session.as_default():
            self.init_op.run()
            self.__saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.__max_to_keep)
            # Loading last save if needed
            if self.__restore:
                ckpt = tf.train.latest_checkpoint(self.__checkPoint_dir)
                if ckpt:
                    self.step = int(ckpt.split('-')[1])
                    self.__saver.restore(self.__session, ckpt)
                    print('Restoring from epoch:{}'.format( self.step))
                    self.step+=1

    def model(self):
        def encoder_net(_image, scope, is_training, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                net = tf.layers.batch_normalization(_image, training=is_training)
                net = slim.conv2d(net, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.conv2d(net, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.conv2d(net, 256, [3, 3], activation_fn=None, scope='conv3')
                net = tf.layers.batch_normalization(net, training=is_training)
                net = tf.nn.relu(net)
                net = slim.conv2d(net, 256, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], [1, 2], scope='pool3')
                net = slim.conv2d(net, 512, [3, 3], activation_fn=None, scope='conv5')
                net = tf.layers.batch_normalization(net, training=is_training)
                net = tf.nn.relu(net)
                net = slim.conv2d(net, 512, [3, 3], scope='conv6')
                net = slim.max_pool2d(net, [2, 2], [1, 2], scope='pool4')
                net = slim.conv2d(net, 512, [2, 2], padding='VALID', activation_fn=None, scope='conv7')
                net = tf.layers.batch_normalization(net, training=is_training)
                net = tf.nn.relu(net)  # CRNN
                cnn_out = tf.squeeze(net, axis=2)

                cell = tf.contrib.rnn.GRUCell(num_units=RNN_UNITS)
                enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=cnn_out,
                                                                         dtype=tf.float32)  # 双向LSTM
                encoder_outputs = tf.concat(enc_outputs, -1)
                return encoder_outputs, enc_state

        def decode(helper, memory, scope, enc_state, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=RNN_UNITS, memory=memory)
                cell = tf.contrib.rnn.GRUCell(num_units=RNN_UNITS)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                                attention_layer_size=RNN_UNITS,
                                                                output_attention=True)
                output_layer = Dense(units=VOCAB_SIZE)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=attn_cell, helper=helper,
                    initial_state=attn_cell.zero_state(dtype=tf.float32, batch_size=self.__batch_size).clone(
                        cell_state=enc_state[0]),
                    output_layer=output_layer)
                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=True, maximum_iterations=27)
                return outputs

        image = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE[1], IMAGE_SIZE[0], 1), name='img_data')
        train_output = tf.placeholder(tf.int64, shape=[None, None], name='train_output')
        target_output = tf.placeholder(tf.int64, shape=[None, None], name='target_output')
        sample_rate = tf.placeholder(tf.float32, shape=[], name='sample_rate')
        train_length = np.array([27] * self.__batch_size, dtype=np.int32)
        train_output_embed, enc_state = encoder_net(image, 'encode_features', self.is_training)
        # vocab_size: 输入数据的总词汇量，指的是总共有多少类词汇，不是总个数，embed_dim：想要得到的嵌入矩阵的维度
        embeddings = tf.get_variable(name='embed_matrix', shape=[VOCAB_SIZE, VOCAB_SIZE])
        output_embed = embedding_ops.embedding_lookup(embeddings, train_output)
        start_tokens = tf.zeros([self.__batch_size], dtype=tf.int64)
        train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(output_embed, train_length,
                                                                           embeddings, sample_rate)
        # 用于inference阶段的helper，将output输出后的logits使用argmax获得id再经过embedding layer来获取下一时刻的输入。
        # start_tokens： batch中每个序列起始输入的token_id  end_token：序列终止的token_id
        # start_tokens: int32 vector shaped [batch_size], the start tokens.
        # end_token: int32 scalar, the token that marks end of decoding.
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=tf.to_int32(start_tokens),
                                                               end_token=1)  # GO,EOS的序号
        train_outputs = decode(train_helper, train_output_embed, 'decode', enc_state)

        pred_outputs = decode(pred_helper, train_output_embed, 'decode', enc_state, reuse=True)
        train_decode_result = train_outputs[0].rnn_output[:, :-1, :]
        pred_decode_result = pred_outputs[0].rnn_output
        mask = tf.cast(tf.sequence_mask(self.__batch_size * [train_length[0] - 1], train_length[0]),
                       tf.float32)
        att_loss = tf.contrib.seq2seq.sequence_loss(train_outputs[0].rnn_output, target_output, weights=mask)

        loss = tf.reduce_mean(att_loss)
        optimizer = tf.train.AdamOptimizer(self.__learn_rate)
        train_op = optimizer.minimize(loss)
        init_op=tf.global_variables_initializer()
        return image,train_output,target_output,train_decode_result,sample_rate, pred_decode_result,loss,train_op,init_op

    def save(self):
        self.__saver.save(
            self.__session,
            os.path.join(self.__checkPoint_dir, 'ckp'),
            global_step=self.step
        )

    # def save_PbModel(self):
    #     output_name=self.__decoded.op.name
    #     #output_name = self.__decoded.name.split(":")[0]
    #     input1_name=self.__inputs.name.split(":")[0]
    #     input2_name = self.__seq_len.name.split(":")[0]
    #     print("模型保存为pb格式，输入节点name：{}，{},输出节点name: {}".format(input1_name,input2_name,output_name))
    #     #constant_graph = graph_util.convert_variables_to_constants(self.__session, self.__session.graph_def, [output_name])
    #     constant_graph=graph_util.convert_variables_to_constants(self.__session,self.__session.graph_def,["SparseToDense"])
    #     with tf.gfile.GFile(self.__model_path+'Model.pb', mode='wb') as f:
    #         f.write(constant_graph.SerializeToString())
    #def PbModel(self):
        # with gfile.FastGFile('Model.pb', 'rb') as f:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(f.read())
        #     sess.graph.as_default()
        #     tf.import_graph_def(graph_def, name='')  # 导入计算图
        #     for i, n in enumerate(graph_def.node):
        #         print("Name of the node - %s" % n.name)
		#
        # # 输入
        # input_x = sess.graph.get_tensor_by_name('Placeholder:0')
        # input_seq_len = sess.graph.get_tensor_by_name('seq_len:0')
        # # 输出
        # op = sess.graph.get_tensor_by_name('SparseToDense:0')