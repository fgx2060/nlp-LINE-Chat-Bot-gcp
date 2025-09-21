# 🧠 自建 NLP 聊天機器人（LINE + GCP 部署）

## 📖 專案簡介
本專案實作了一個基於 **GPT 模型** 的 NLP 聊天機器人，涵蓋：
1. **資料處理與詞典建立**
2. **GPT 模型訓練**
3. **互動式測試**
4. **LINE Bot 與 GCP 部署**

最終目標是讓使用者能透過 **LINE Bot 與訓練好的模型進行對話**

---

## ✨ 功能特色
- **資料清理**：正規化對話資料，支援中英文與特殊標記。
- **詞典建立**：`Word2Sequence` 建立字 ↔ 索引轉換，支援 PAD、UNK、EOS。
- **GPT 模型訓練**：
  - 自製資料集 (`train2.txt → dataset.txt`)
  - 搭配 AdamWarmup 與 Label Smoothing CrossEntropy
  - 產生 `model.pth.rar`、`model_last.pth.rar`
- **互動測試**：支援 CLI 測試，輸入訊息即可獲得模型回覆。
- **LINE Bot 整合**：透過 `token_and_secret.py` 與 Flask Webhook，將模型部署到 GCP 並連接 LINE Bot。

## 建構流程參考 :  
 - 類神經網路與深度學習-自建 NLP 聊天機器人（LINE + GCP 部署）.pdf
---
## 📂 專案結構
```
nlp-linebot-gcp/
├─ data/
│  ├─ train.txt              # 原始訓練資料
│  ├─ train2.txt             # 第二版訓練資料
│  ├─ dataset.txt            # 清理後的資料
│  └─ ws.pkl                 # 詞典檔 (word2seq 輸出)
│
├─ src/
│  ├─ sol_data.py            # 資料清理與正規化 (train2 → dataset)
│  ├─ word2seq.py            # 詞典建立 (Word2Sequence)
│  ├─ dataset.py             # Dataset 定義，提供 DataLoader
│  ├─ gpt_model.py           # GPT 模型架構
│  ├─ utils.py               # 訓練工具 (AdamWarmup, LossWithLS, get_acc)
│  ├─ train.py               # 模型訓練
│  ├─ test.py                # 命令列測試（終端機互動）
│  └─ chat.py                # LINE Bot 主程式（部署用）
│
├─ deployment/
│  ├─ config.py              # 設定檔（參數、環境變數）
│  ├─ token_and_secret.py    # LINE Bot Token / Secret 
│  ├─ requirements.txt       # 依賴套件清單
│  └─ run_ngrok.sh           # (可新增) Linux 上背景執行 Ngrok
│
├─ models/
│  ├─ model.pth.rar          # 訓練後的模型
│  └─ model_last.pth.rar     # 最終模型
│
├─ notebooks/
│  ├─ change.ipynb           # 資料前處理 Notebook（含簡繁轉換、初步測試）
│
├─ docs/
│  └─ 類神經網路與深度學習-自建 NLP 聊天機器人（LINE + GCP 部署）.pdf
│
└─ README.md                 # 專案說明
```
- 備註 : 資料及漢模型檔案太大就沒有放上來了
## 影片https://youtu.be/rKzG02KMkTE













原始資料集:https://drive.google.com/file/d/1nEuew_KNpTMbyy7BO4c8bXMXN351RCPp/view
