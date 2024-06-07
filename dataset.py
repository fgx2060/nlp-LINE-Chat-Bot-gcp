# 建立資料集
import torch
import pickle
import config
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from word2seq import Word2Sequence

word_sequence = pickle.load(open(config.word_sequence_dict, "rb"))  # 詞典

class ChatDataset(Dataset):
    def __init__(self):
        super(ChatDataset, self).__init__()

        # 讀取內容
        data_path = config.data_path_txt
        self.data_lines = open(data_path, encoding='utf-8').readlines()

    # 獲取對應索引的問答
    def __getitem__(self, index):
        input = self.data_lines[index].strip().split()[:-1]
        target = self.data_lines[index].strip().split()[1:]
        # 若為空則默認讀取下一條
        if len(input) == 0 or len(target) == 0:
            input = self.data_lines[index + 1].split()[:-1]
            target = self.data_lines[index + 1].split()[1:]
        # 此處句子的長度如果大於max_len，那麼應該返回max_len
        return input, target, len(input), len(target)

    # 獲取數據長度
    def __len__(self):
        return len(self.data_lines)

# 整理數據————數據集處理方法
def collate_fn(batch):
    # 排序
    batch = sorted(batch, key=lambda x: x[2], reverse=True)  # 輸入長度排序
    input, target, input_length, target_length = zip(*batch)

    max_len = max(input_length[0], target_length[0])  # 這裡只需要固定每個batch裏面的樣本長度一致就好，並不需要整個數據集的所有樣本長度一致

    # 詞變成詞向量，並進行padding的操作
    input = torch.LongTensor([word_sequence.transform(i, max_len=max_len, add_eos=False) for i in input])
    target = torch.LongTensor([word_sequence.transform(i, max_len=max_len, add_eos=False) for i in target])

    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)

    return input, target

print("數據集裝載...")
data_loader = DataLoader(dataset=ChatDataset(), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
                         drop_last=True)

'''
if __name__ == '__main__':
    for idx, (input, target) in enumerate(data_loader):
        print(idx)
        print(input)
        print(target)
'''
