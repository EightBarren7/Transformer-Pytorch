import os
import numpy as np
from config import *
from dataloader import en_word_dict, cn_index_dict
"""
这里使用了最简单的贪心解码策略，没有对准确率作任何优化
因此实际效果并不佳，理解模型并完成整个工程是本代码的重点
"""
filepath = os.path.join(checkpoint_path, 'checkpoint_best_model.pth')
model = torch.load(filepath)
while 1:
    sent = input('请输入要翻译的英文\n')
    sent = sent.split()
    BOS = en_word_dict['BOS']
    EOS = en_word_dict['EOS']

    src = [[BOS] + [en_word_dict[w] for w in sent] + [EOS]]
    sent_input = torch.LongTensor(np.array(src)).to(device)
    with torch.no_grad():
        model.eval()
        memory = model.encode(sent_input)
        start_symbol = BOS
        end_symbol = EOS
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(sent_input.data)
        for i in range(max_len-1):
            out = model.decode(ys, sent_input, memory)
            prob = out[:,-1]
            next_word = torch.argmax(prob)
            if next_word == end_symbol:
                break
            ys = torch.cat([ys, torch.ones(1, 1).type_as(sent_input.data).fill_(next_word)], dim=1)
    ys = ys.squeeze(0)
    ys = ys.cpu().numpy()
    translations = [cn_index_dict[index] for index in ys]
    print(translations)