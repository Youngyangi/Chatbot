from flask import Flask, request, render_template
from task_seq2seq_autotitle import *
from util.humanic_module import get_answer, get_pattern
import jieba.analyse, jieba.posseg
import jieba
import pickle
import json
import wikipedia


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/chat/', methods=['POST', 'GET'])
def chat():
    if request.method == 'POST':
        data = json.loads(request.get_data(as_text=True))
        # data = request.get_data(as_text=True)
        # if request.form['text']:
        if 'text' in data:
            text = data['text']
            if get_answer(text.strip(), humanic_dict) != -1:
                return get_answer(text.strip(), humanic_dict)
            is_query = 1-semantic_layer([cut(text)], semantic_model, bow)
            if is_query:
                entity = ner_extract(text)
                general = '不好意思，小通现在还不会这个'
                return general if not entity else f"小通知道你想查询{entity}的股票价格，不过小通现在还不会。"
            else:
                response = autotitle.generate(text)
                if response == "==":
                    response = "= = 请原谅，小通不明白你说的是什么。"
                return response
    title = request.args.get('title', 'Default')
    return render_template('chat.html', title=title)


@app.route('/wiki/', methods=['GET', 'POST'])
def wiki():
    if request.method == 'POST':
        data = json.loads(request.get_data(as_text=True))
        wikipedia.set_lang("zh")
        keyword = wikipedia.page(data)
        response = keyword.summary
        return response

    title = request.args.get('title', 'Default')
    return render_template('chat.html', title=title)



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


def cut(text):
    return ' '.join(jieba.cut(text))


def init_semantic():
    with open("semantic.pkl", 'rb') as pk:
        model = pickle.load(pk)
    with open("bow.pkl", 'rb') as pk:
        bow = pickle.load(pk)

    return model, bow


def semantic_layer(input, model, bow):
    id = bow.transform(input)
    res = model.predict(id)
    return res[0]


def ner_extract(sent):
    task = ['股票', '股票价格']
    entity = jieba.analyse.extract_tags(sent, allowPOS=['nz', 'nrt', 'a'])
    if not entity:
        words = jieba.posseg.cut(sent)
        words = [(x, y) for x, y in words]
        for x in task:
            for y in words:
                if x in y:
                    ind = words.index(y)
                    if words[ind-1][1] == 'v':
                        break
                    entity = words[ind-1]
    return entity


semantic_model, bow = init_semantic()

humanic_dict = get_pattern()

# 基本参数
maxlen = 150

# bert配置
config_path = r'E:\Pretrained_Model\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'E:\Pretrained_Model\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'E:\Pretrained_Model\chinese_L-12_H-768_A-12\vocab.txt'

token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)
model.load_weights('./ckpt/best_model.weights')

model.summary()

autotitle = AutoTitle(start_id=None,
                      end_id=tokenizer._token_end_id,
                      maxlen=100)

just_show()

app.run(debug=False)
