import pickle
import config
import torch
import torch.utils.data
from gpt_model import *
from dataset import data_loader
from utils import AdamWarmup, LossWithLS, get_acc
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 初始化SummaryWriter，用於記錄訓練過程中的指標
summaryWriter = SummaryWriter("logs/log2")

# 從配置文件中讀取模型參數
emb_dim = config.emb_dim
max_pos = config.max_pos
heads = config.heads
d_k = config.d_k
d_v = config.d_v
num_layers = config.num_layers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 嘗試使用第二個GPU，如果無法使用則使用CPU
epochs = config.epochs

# 加載詞彙映射字典
word_map = pickle.load(open(config.word_sequence_dict,"rb"))
print(len(word_map.dict))

# 如果配置中指定為新訓練，則創建一個新的GPT模型實例並初始化優化器
if config.load == False:
    gpt = GPT(vocab_size=len(word_map.dict), d_model=emb_dim, max_pos=max_pos, n_heads= heads, d_k=d_k, d_v=d_v, n_layers=num_layers).to(device)
    adam_optimizer = torch.optim.Adam(gpt.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)  # 使用Adam優化器
    epoch_start = 0
else:  # 如果配置中指定為繼續訓練，則加載之前的模型和優化器
    checkpoint = torch.load('model.pth.rar')
    gpt = checkpoint['gpt']
    adam_optimizer = checkpoint['adam_optimizer']
    epoch_start = checkpoint['epoch'] + 1

# 定義損失函數
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

# 定義訓練函數
def train(train_loader, gpt, criterion, epoch):
    gpt.train()
    sum_loss = 0
    count = 0
    sum_acc = 0

    for i, (question, reply) in enumerate(train_loader):
        torch.cuda.empty_cache()  # 釋放緩存空間

        samples = question.shape[0]

        # 將輸入資料移動到指定設備
        question = question.to(device)
        reply = reply.to(device)

        # 獲取GPT模型的輸出
        out = gpt(question)

        # 計算損失
        loss = criterion(out.view(-1, out.size(-1)), reply.view(-1))
        acc = get_acc(out, reply)
        
        # 反向傳播計算梯度
        adam_optimizer.zero_grad()
        loss.backward()
        adam_optimizer.step()

        sum_loss += float(loss.item()) * samples
        sum_acc += acc.item() * samples
        count += samples
        
        if i % 1 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}\tAcc: {:.3f}%".format(epoch, i, len(train_loader), sum_loss/count, (sum_acc/count)*100))  # 輸出累計情況下的平均一個詞的損失

    return sum_loss/count
            
print("訓練...")    
loss_max = 10000000000
for epoch in range(epoch_start, epochs):
    loss = train(data_loader, gpt, criterion, epoch)
    
    # TensorBoard實時監控
    summaryWriter.add_scalars('epoch_metric', {'epoch_loss': loss }, epoch)

    if loss_max > loss:  # 選擇性保存
        print("保存輪數：",epoch)
        loss_max = loss
    
        state = {'epoch': epoch, 'gpt': gpt, 'adam_optimizer': adam_optimizer}

        torch.save(state, 'model.pth.rar')  # 記下每次最好的結果（為了防止中斷程序後，什麼都沒保存）
    
    if epoch == epochs-1:  # 保存最後的結果
        torch.save(state, 'model_last.pth.rar')
