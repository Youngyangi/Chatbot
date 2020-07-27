#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933

from __future__ import print_function
import glob
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder




# 训练样本。THUCNews数据集，每个样本保存为一个txt。
# txts = glob.glob('data/THUCNews/*/*.txt')


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            try:
                query, answer = l.strip().split('\t')
                D.append((answer, query))
            except:
                pass
    return D




class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (title, content) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(content,
                                                      title,
                                                      max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


# class data_generator(DataGenerator):
#     """数据生成器
#     """
#     def __iter__(self, random=False):
#         batch_token_ids, batch_segment_ids = [], []
#         for is_end, txt in self.sample(random):
#             text = open(txt, encoding='utf-8').read()
#             text = text.split('\n')
#             if len(text) > 1:
#                 title = text[0]
#                 content = '\n'.join(text[1:])
#                 token_ids, segment_ids = tokenizer.encode(content,
#                                                           title,
#                                                           max_length=maxlen)
#                 batch_token_ids.append(token_ids)
#                 batch_segment_ids.append(segment_ids)
#             if len(batch_token_ids) == self.batch_size or is_end:
#                 batch_token_ids = sequence_padding(batch_token_ids)
#                 batch_segment_ids = sequence_padding(batch_segment_ids)
#                 yield [batch_token_ids, batch_segment_ids], None
#                 batch_token_ids, batch_segment_ids = [], []




class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, max_length=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids], topk)  # 基于beam search
        return tokenizer.decode(output_ids)




def just_show():
    s1 = "在吗"
    s2 = "你叫什么名字"
    s3 = "我想和你聊天"
    s4 = "今天天气怎么样"
    s5 = "谁是世界上最美的女人"
    for s in [s1, s2, s3, s4, s5]:
        print(u'生成对话:', autotitle.generate(s), '原文本长度：', len(s), '字')


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./ckpt/best_model.weights')
        # 演示效果
        just_show()


if __name__ == '__main__':

    # 基本参数
    maxlen = 150
    batch_size = 16
    epochs = 2

    # bert配置
    config_path = r'E:\Pretrained_Model\chinese_L-12_H-768_A-12\bert_config.json'
    checkpoint_path = r'E:\Pretrained_Model\chinese_L-12_H-768_A-12\bert_model.ckpt'
    dict_path = r'E:\Pretrained_Model\chinese_L-12_H-768_A-12\vocab.txt'

    txts = load_data('data/chat/xiaohuangji.tsv')

    steps_per_epoch = len(txts) // batch_size

    # 加载并精简词表，建立分词器
    token_dict, keep_tokens = load_vocab(
        dict_path=dict_path,
        simplified=True,
        startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    tokenizer = Tokenizer(token_dict, do_lower_case=True)

    evaluator = Evaluate()
    train_generator = data_generator(txts, batch_size)

    model = build_transformer_model(
        config_path,
        checkpoint_path,
        application='unilm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    )

    model.summary()

    autotitle = AutoTitle(start_id=None,
                          end_id=tokenizer._token_end_id,
                          maxlen=100)

    # 交叉熵作为loss，并mask掉输入部分的预测
    y_true = model.input[0][:, 1:]  # 目标tokens
    y_mask = model.input[1][:, 1:]
    y_pred = model.output[:, :-1]  # 预测tokens，预测与目标错开一位
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

    model.add_loss(cross_entropy)
    model.compile(optimizer=Adam(1e-5))

    model.load_weights('./ckpt/best_model.weights')
    just_show()
    model.save('chat_model')


    # tf.saved_model.save(model, "/saved_model/chatbot/")

    # model.fit_generator(train_generator.forfit(),
    #                     steps_per_epoch=steps_per_epoch,
    #                     epochs=epochs,
    #                     callbacks=[evaluator])

else:
    pass
