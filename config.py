import torch

checkpoint_path = './result'
batch_size = 128
d_k = 64  # k矩阵的维度
d_v = 64  # v矩阵的维度
d_model = 512  # 每一个单词的embedding维度
n_heads = 8  # 多头自注意力的头数量
d_ff = 2048
n_layers = 6
lr = 0.001
num_epochs = 100
max_len = 64  # 预测的单词最大长度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
