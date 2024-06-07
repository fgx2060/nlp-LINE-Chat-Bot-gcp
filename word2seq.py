#產生詞表
#建構文字序列化與反序列化方法（文字轉數字）
#import matplotlib.pyplot as plt
import pickle
import config
from tqdm import tqdm

class Word2Sequence():
    PAD_TAG = "<PAD>" #填充編碼
    UNK_TAG = "<UNK>" #未知編碼
    EOS_TAG = "<EOS>" #句子結尾

    #上面四種情況的對應編號
    PAD = 0
    UNK = 1
    EOS = 2

    def __init__(self):

        #文字－標號字典
        self.dict = {
            self.PAD_TAG :self.PAD,
            self.UNK_TAG :self.UNK,
            self.EOS_TAG :self.EOS
        }
        #詞頻統計
        self.count = {}
        self.fited = False #是否統計過字典了

    #以下兩個轉換都不包括'\t'
    #文字轉標號碼（針對單字）
    def to_index(self,word):
        """word -> index"""
        assert self.fited == True,"必須先進行fit操作"
        return self.dict.get(word,self.UNK) #无这个词则用未知代替

    #標號轉文字（針對單字）
    def to_word(self,index):
        """index -> word"""
        assert self.fited == True, "必須先進行fit操作"
        if index in self.inversed_dict:
            return self.inversed_dict[index]
        return self.UNK_TAG

    # 取得字典長度
    def __len__(self):
        return len(self.dict)

    #統計詞頻生成詞典
    def fit(self, sentence):
        """
        :param sentence:[word1,word2,word3]
        """
        for a in sentence:
            if (a != '<EOS>'):
                if (a not in self.count):
                    self.count[a] = 0
                self.count[a] += 1

        self.fited = True

    def build_vocab(self, min_count=config.min_count, max_count=None, max_feature=None):

        # 限定統計詞頻範圍
        if min_count is not None:
            self.count = {k: v for k, v in self.count.items() if v >= min_count}
        if max_count is not None:
            self.count = {k: v for k, v in self.count.items() if v <= max_count}

        # 給對應詞進行編號
        if isinstance(max_feature, int): #是否限製字典的詞數
            #詞頻由大到小排序
            count = sorted(list(self.count.items()), key=lambda x: x[1])
            if max_feature is not None and len(count) > max_feature:
                count = count[-int(max_feature):]
            for w, _ in count:
                self.dict[w] = len(self.dict)
        else: #按字典序(方便debug查看)
            for w in sorted(self.count.keys()):
                self.dict[w] = len(self.dict)

        # 準備一个index->word的字典
        self.inversed_dict = dict(zip(self.dict.values(), self.dict.keys()))

        #debug專用
        f_debug_word = open("C:/python/PY4/debug_word2.txt","w",encoding='utf-8')
        t = 0
        for key,_ in self.dict.items():
            t = t + 1
            if t > 3:
                f_debug_word.write(key+"★ "+str(self.count[key]) + "\n") #使用★ 區分是為了防止其中的詞語包含分隔符，對我們後續的操作不利

        f_debug_word.close()

    def transform(self, sentence,max_len=None,add_eos=True):
        """
        實現把句子轉化为向量
        :param sentence:
        :param max_len:
        :return:
        """
        assert self.fited == True, "必須先進行fit操作"

        r = [self.to_index(i) for i in sentence]
        if max_len is not None: #限定長度
            if max_len>len(sentence):
                if add_eos:
                    #添加结束符與填充符達到一定長度
                    r+=[self.EOS]+[self.PAD for _ in range(max_len-len(sentence)-2)]
                else: #添加填充符達到一定長度
                    r += [self.PAD for _ in range(max_len - len(sentence)-1)]
            else:
                if add_eos:
                    r = r[:max_len-2]
                    r += [self.EOS]
                else:
                    r = r[:max_len-1]
        else:
            if add_eos:
                r += [self.EOS]

        return r

    def inverse_transform(self,indices):
        """
        實現從句子向量 轉換為 字（文字）
        :param indices: [1,2,3....]
        :return:[word1,word2.....]
        """
        sentence = []
        for i in indices:
            word = self.to_word(i)
            sentence.append(word)
        return sentence



#初始
word_sequence = Word2Sequence()
#词语导入
data_path = config.data_path_txt
for line in tqdm(open(data_path,encoding='utf-8').readlines()):
    word_sequence.fit(line.strip().split())


print("生成词典...")
word_sequence.build_vocab(min_count=None,max_count=None,max_feature=None)#这里不限制词典词语数目
print("词典大小：",len(word_sequence.dict))
pickle.dump(word_sequence,open(config.word_sequence_dict,"wb"))
