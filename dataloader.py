import numpy as np
from nltk import word_tokenize
from collections import Counter
import torch.utils.data as Data
from config import *

train_path = 'data/train.txt'
test_path = 'data/dev.txt'
PAD = 0
UNK = 1


def seq_padding(X, padding=0):
    """
    对一个batch批次(以单词id表示)的数据进行padding填充对齐长度
    """
    # 计算该批次数据各条数据句子长度
    L = [len(x) for x in X]
    # 获取该批次数据最大句子长度
    ML = max(L)
    # 对X中各条数据x进行遍历，如果长度短于该批次数据最大长度ML，则以padding id填充缺失长度ML-len(x)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class PreparedData:
    def __init__(self, train_file, test_file):
        # 读取数据并分词
        self.train_en_data, self.train_cn_data = self.load_data(train_file)
        self.test_en_data, self.test_cn_data = self.load_data(test_file)
        # 构建词表
        self.en_word_dicts, self.en_index_dicts, self.en_total_words = self.build_dict(self.train_en_data)
        self.cn_word_dicts, self.cn_index_dicts, self.cn_total_words = self.build_dict(self.train_cn_data)
        # 将文本数字化
        self.train_en, self.train_cn = self.word2ID(self.train_en_data, self.train_cn_data, self.en_word_dicts,
                                                    self.cn_word_dicts)
        self.test_en, self.test_cn = self.word2ID(self.test_en_data, self.test_cn_data, self.en_word_dicts,
                                                  self.cn_word_dicts)

    def load_data(self, path):
        """
        将数据读取分句分词并构建成包含开始符[BOS]和结束符[EOS]的列表
        原始数据：
        Anyone can do that.	任何人都可以做到。……
        返回数据：
        en = [['BOS', 'Anyone', 'can', 'do', 'that', '.', 'EOS'], [……], ……]
        cn = [['BOS', '任', '何', '人', '都', '可', '以', '做', '到', '。', 'EOS'], [……], ……]
        """
        en = []
        cn = []
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                list_content = line.split('\t')
                en.append(['BOS'] + word_tokenize(list_content[0]) + ['EOS'])
                cn.append(['BOS'] + word_tokenize(' '.join(list_content[1])) + ['EOS'])
        return en, cn

    def build_dict(self, sentences, max_word=50000):
        """
        传入分完词后的列表数据，构建词典
        返回构建好的词典和总词数
        此时只要通过词典便可以将单词转换为数字，通过索引表将数字变回单词
        如：word_dicts['PAD'] = 0, index_dicts[0] = 'PAD'
        """
        word_count = Counter()  # 对所有单词计数
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1
        ls = word_count.most_common(max_word)  # 只保留频率最高的前max_word个词，默认为50000
        total_words = len(ls) + 2  # 统计总词数并加上PAD和UNK
        # 按单词出现频率生成词典 word to index
        word_dicts = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dicts['PAD'] = PAD
        word_dicts['UNK'] = UNK
        # 再反向构建一个索引表 index to word
        index_dicts = {index: w for w, index in word_dicts.items()}
        return word_dicts, index_dicts, total_words

    def word2ID(self, en, cn, en_dicts, cn_dicts):
        """
        将分完词后的列表数据利用词典转换为数字
        原始数据：
        en = [['BOS', 'Anyone', 'can', 'do', 'that', '.', 'EOS'], [……], ……]
        cn = [['BOS', '任', '何', '人', '都', '可', '以', '做', '到', '。', 'EOS'], [……], ……]
        返回数据：
        注意里面的数据只是为了明白数据格式举例用，并非真实值
        en = [[0, 1, 2, 3, 4, 5, 6], [……], ……]
        cn = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [……], ……]
        """
        out_en_ids = [[en_dicts.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dicts.get(w, 0) for w in sent] for sent in cn]
        return out_en_ids, out_cn_ids

    def get_len(self):
        """
        返回词典总词数
        """
        return self.en_total_words, self.cn_total_words

    def get_word_dict(self):
        """
        返回词典
        """
        return self.en_word_dicts, self.cn_word_dicts

    def get_index_dict(self):
        """
        返回索引表
        """
        return self.en_index_dicts, self.cn_index_dicts

    def get_train_data(self):
        """
        返回经过padding后的训练数据
        padding是为了保证每一句话的长度相等，便于后续操作
        如果不明白，请学习参考transformer中的padding操作的实际过程和作用
        这里输出三个数据：
        train_en：encoder的输入，要翻译的英文数据
        train_cn：decoder的输入，翻译后的中文数据
        train_cn：标签，翻译后的中文数据，和decoder的输出作loss并反向传播
        """
        train_en = seq_padding(self.train_en)
        train_cn = seq_padding(self.train_cn)
        return torch.LongTensor(train_en), torch.LongTensor(train_cn), torch.LongTensor(train_cn)

    def get_test_data(self):
        """
        返回经过padding后的测试数据
        """
        test_en = seq_padding(self.test_en)
        test_cn = seq_padding(self.test_cn)
        return torch.LongTensor(test_en), torch.LongTensor(test_cn), torch.LongTensor(test_cn)


class MyDataSet(Data.Dataset):
    """
    构建dataset方便传入pytorch的dataloader模块
    """

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


data = PreparedData(train_path, test_path)

train_enc_inputs, train_dec_inputs, train_dec_outputs = data.get_train_data()
train_dec_inputs = train_dec_inputs[:, :-1]
train_dec_outputs = train_dec_outputs[:, 1:]
"""
enc_inputs: [total_words,max_len]
dec_inputs: [total_words,max_len-1]
dec_outputs: [total_words,max_len-1]
decoder的输入要将最后的[EOS]去掉
标签要将最开始的[BOS]去掉
这一步操作其实是对应解码器的贪心搜索
我们给解码器输入一个[BOS]开始标志，解码器自动预测下一个词
直到我们给解码器最后一个单词(注意不是[EOS])，解码器预测出停止标志[EOS]
因此实际上我们给解码器的输入并不是完整的句子，而是['BOS','W1','W2',……,'Wfinal']，并没有[EOS]
而解码器的输出为['W1','W2',……,'Wfinal','EOS']，并没有[BOS]
因此标签数据也要去除[BOS]，这样在计算loss时再能保持一致
"""

test_enc_inputs, test_dec_inputs, test_dec_outputs = data.get_test_data()
test_dec_inputs = test_dec_inputs[:, :-1]
test_dec_outputs = test_dec_outputs[:, 1:]

enc_vocab_size, dec_vocab_size = data.get_len()

en_word_dict, cn_word_dict = data.get_word_dict()
en_index_dict, cn_index_dict = data.get_index_dict()

train_loader = Data.DataLoader(MyDataSet(train_enc_inputs, train_dec_inputs, train_dec_outputs), batch_size, True)
test_loader = Data.DataLoader(MyDataSet(test_enc_inputs, test_dec_inputs, test_dec_outputs), batch_size, True)