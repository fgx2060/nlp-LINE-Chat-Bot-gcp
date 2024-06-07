import re
from tqdm import tqdm
import zhconv
import config
 
#處理重複符號的表達，如替換多個重複符號
def delete_repeat(s):
    #註解掉的是英文的表達
    #s = re.sub('[!]+','!', s)
    #s = re.sub('[?]+','?', s)
    #s = re.sub('[,]+',',', s)
    #s = re.sub('[:]+',':', s)
    #s = re.sub('[;]+',';', s)
    s = re.sub('[，]+','，', s)
    s = re.sub('[！]+','！', s)
    s = re.sub('[？]+','？', s)
    s = re.sub('[：]+','：', s)
    s = re.sub('[；]+','；', s)
    s = re.sub('[。]+','。', s)
    s = re.sub('[、]+','、', s)
    return s
 
with open('C:/python/PY4/train2.txt','r',encoding='utf-8') as f: #開啟原始資料集
    lines = f.readlines()
 
train_datas = []
temp_data = ''
#每個多輪對話中使用'<EOS>'將其劃分
for line in tqdm(lines):
 
    if line!='\n':
        line = line.strip() #去除前導後方空格
        #英文標點符號置換為中文標點符號
        line = line.replace('!','！')
        line = line.replace('?','？')
        line = line.replace(',','，')
        line = line.replace('.','。')
        line = line.replace(':','：')
        line = line.replace(';','；')
        line = zhconv.convert(line, 'zh-tw') #轉為繁體字
        line = " ".join(line)
        temp_data+=(line+' <EOS> ')
    else:
        if len(temp_data.split()) <= config.max_len: #限制長度
            train_datas.append(temp_data)
        temp_data=''
 
with open(config.data_path_txt,'w',encoding='utf-8') as f: #將處理後的資料保存在另一個檔案中
    for train_data in train_datas:
        f.write(train_data+'\n')