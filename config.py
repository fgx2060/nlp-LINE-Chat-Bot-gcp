# 數據文件的路徑
data_path_txt = 'C:/python/PY4/dataset.txt'

# 詞典序列的路徑
word_sequence_dict = 'C:/python/PY4/ws.pkl'

# 是否加載已有模型或數據
load = True

# 文本序列的最大長度
max_len = 100

# 批處理大小
batch_size = 32

# 詞頻的最小值，過濾掉低於此頻率的詞
min_count = 3

# 詞頻的最大值，過濾掉高於此頻率的詞
max_count = 5000

# 詞嵌入的維度（向量的長度）
emb_dim = 768

# 多頭注意力機制中的頭數
heads = 8

# 模型的層數（例如Transformer的編碼器或解碼器層數）
num_layers = 6

# 訓練的輪數
epochs = 500

# 位置編碼的最大值
max_pos = 1800

# Query和Key向量的維度（多頭注意力機制中的參數）
d_k = 64

# Value向量的維度（多頭注意力機制中的參數）
d_v = 64
