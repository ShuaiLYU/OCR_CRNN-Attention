import  tensorflow as tf
import numpy as np
import shutil
import  os
from data_manager import DataManager
from  acrnn import  ACRNN
from config import  *


class Agent(object):
	def __init__(self,param):

		self.__sess=tf.Session()
		self.__Param=param
		self.init_datasets()  #初始化数据管理器
		self.model=ACRNN(self.__sess,self.__Param) #建立模型

	def run(self):
		if self.__Param["mode"] is "training":
			self.train()
		elif self.__Param["mode"] is "testing":
			self.test()
		elif self.__Param["mode"] is "savePb":
			print(" this  mode is incomplete ")
		else:
			print("got a unexpected mode ,please set the mode  'training', 'testing' or 'savePb' ")

	def init_datasets(self):
		self.data_list, self.data_size = self.listData(self.__Param["data_dir"])
		if self.__Param["mode"] is "training":
			if self.__Param["valid_ratio"] > 1.0 or self.__Param["valid_ratio"] < 0:
				raise Exception('Incoherent ratio!')
			valid_offset = int(len(self.data_list) * (1 - self.__Param["valid_ratio"]))
			self.data_list_train = self.data_list[0:valid_offset]
			self.DataManager_train = DataManager(self.data_list_train, self.__Param)
			self.data_list_valid = self.data_list[valid_offset:-1]
			self.DataManager_valid = DataManager(self.data_list_valid, self.__Param)
		elif self.__Param["mode"] is "testing":
			self.data_list_test = self.data_list
			self.DataManager_test = DataManager(self.data_list_test, self.__Param)
		elif self.__Param["mode"] is "savePb":
			pass
		else:
			raise Exception('got a unexpected  mode ')

	def train(self):
		with self.__sess.as_default():
			print('Training')
			for i in range(self.model.step, self.__Param["epochs_num"] + self.model.step):
				iter_loss = 0
				for batch in range(self.DataManager_train.number_batch):
					img_batch, target_input_batch, target_out_batch, filename_batch = self.__sess.run(self.DataManager_train.next_batch)
					_, loss_value = self.__sess.run(
						[self.model.train_op, self.model.loss],
						feed_dict={self.model.image:img_batch,
							self.model.train_output: target_input_batch,
							self.model.target_output: target_out_batch,
							self.model.sample_rate: np.min([1.,0.2*self.model.step+0.2])})
					iter_loss += loss_value

				print('epoch:[{}]  loss: {}'.format(self.model.step, iter_loss))
				#验证
				if i % self.__Param["valid_frequency"] == 0 and i>0:
					self.valid()
				#保存模型
				if i % self.__Param["save_frequency"] == 0 or i==self.__Param["epochs_num"] + self.model.step-1:
					self.model.save()

				self.model.step += 1

	def test(self):
		b_saveNG=self.__Param["b_saveNG"]
		NG_path = self.DataManager_test.data_dir + '_NG'
		if b_saveNG:
			# 创建空txt文档保存错误样本的路径
			if not os.path.exists(NG_path):
				os.mkdir(NG_path)
		with self.__sess.as_default():
			count = 0
			r = 0
			print('testing')
			for batch in range(self.DataManager_test.number_batch):
				img_batch, target_input_batch, target_out_batch, filename_batch = self.__sess.run(self.DataManager_test.next_batch)
				val_predict = self.__sess.run(self.model.pred_decode_result, feed_dict={self.model.image: img_batch})
				val_predict = self.DataManager_test.int2label(np.argmax(val_predict, axis=2))
				for i, filename in enumerate(filename_batch):
					filename=str(filename).split("'")[-2]
					count += 1
					label = self.DataManager_test.get_label(str(filename_batch[i]))
					predict =val_predict[i]

					if label == predict:
						r += 1
					else:
						if b_saveNG:

							wrong_example = os.path.join(self.DataManager_test.data_dir,filename)
							object_path = os.path.join(NG_path, filename)
							shutil.copyfile(wrong_example, object_path)
			acc = r / count
			print("Testing ,count:{},acc:{}".format(count, acc))
			if b_saveNG:
				print("错误样本保存到文件：{}".format(NG_path))

	def valid(self):
		with self.__sess.as_default():
			count = 0
			r = 0
			print('validation...')
			for batch in range(self.DataManager_valid.number_batch):
				img_batch, target_input_batch, target_out_batch, filename_batch = self.__sess.run(
					self.DataManager_valid.next_batch)
				val_predict =  self.__sess.run(self.model.pred_decode_result, feed_dict={self.model.image: img_batch})
				train_predict =  self.__sess.run(self.model.pred_decode_result,
										 feed_dict={self.model.image: img_batch,
													self.model.train_output: target_input_batch,
													self.model.target_output: target_out_batch,
													self.model.sample_rate: np.min([1., 0.2 * self.model.step + 0.2])})
				val_predict = self.DataManager_valid.int2label(np.argmax(val_predict, axis=2))
				train_predict = self.DataManager_valid.int2label(np.argmax(train_predict, axis=2))

				for i, y in enumerate(filename_batch):
					count += 1
					label = self.DataManager_valid.get_label(str(filename_batch[i]))
					predict = val_predict[i]
					train_pre = train_predict[i]
					if label == predict:
						r += 1
					else:
						pass
			acc = r / count
			print("validation ,count:{},acc:{}".format(count, acc))


	#列出文件当前目录下所有文件
	def listData(self,data_dir):
		data_list=os.listdir(data_dir)
		data_list=[x[2] for x in os.walk(data_dir)][0]
		data_size=len(data_list)
		return data_list,data_size

