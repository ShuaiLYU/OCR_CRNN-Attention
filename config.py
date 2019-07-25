#一些宏定义
START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
VOCAB = {'<GO>': 0, '<EOS>': 1, '<UNK>': 2,'<PAD>':3}#分别表示开始，结束，未出现的字符
VOC_IND={}
charset='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
for i in range(len(charset)):
	VOCAB[charset[i]]=i+4
for key in VOCAB:
	VOC_IND[VOCAB[key]]=key
MAX_LEN_WORD=27#标签的最大长度，以PAD
VOCAB_SIZE = len(VOCAB)
RNN_UNITS = 256
IMAGE_SIZE=[32,100]


