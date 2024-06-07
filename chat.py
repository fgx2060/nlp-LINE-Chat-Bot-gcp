from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from token_and_secret import token, secret

import torch
import pickle
import config
from word2seq import Word2Sequence
import json
import logging

app = Flask(__name__)

# Setup logger
logging.basicConfig(level=logging.INFO)

# Initialize LineBotApi and WebhookHandler with your token and secret
line_bot_api = LineBotApi(token)
handler = WebhookHandler(secret)

# Load GPT model and word map
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word_map = pickle.load(open(config.word_sequence_dict, "rb"))  # 词典
checkpoint = torch.load('model.pth.rar')
model = checkpoint['gpt']
model.eval()

def generate_response(input_text):
    sentence = list(input_text.strip()) + ['<EOS>']
    if len(sentence) > 100:
        t_index = sentence.index('<EOS>')
        sentence = sentence[t_index + 1:]
    
    sentence_vec = word_map.transform(sentence, max_len=None, add_eos=False)  # 词转为标号
    dec_input = torch.LongTensor(sentence_vec).to(device).unsqueeze(0)
    
    terminal = False
    start_dec_len = len(dec_input[0])
    while not terminal:
        if len(dec_input[0]) - start_dec_len > 100:
            next_symbol = word_map.dict['<EOS>']
            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
            break
        
        projected = model(dec_input)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == word_map.dict['<EOS>']:
            terminal = True
        
        dec_input = torch.cat(
            [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
    
    out = dec_input.squeeze(0)
    out = word_map.inverse_transform(out.tolist())
    
    eos_indexs = [i for i, x in enumerate(out) if x == "<EOS>"]
    answer = out[eos_indexs[-2] + 1:-1] if len(eos_indexs) > 1 else out[:-1]
    return "".join(answer)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)  # Get the request body as text
    signature = request.headers.get('X-Line-Signature', '')  # Get X-Line-Signature header value
    
    try:
        # Handle webhook body and signature
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)  # If signature is invalid, return 400 Bad Request
    
    try:
        # Parse the request body to JSON
        json_data = json.loads(body)
        # Get the message text and reply token
        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        
        # Generate response using GPT model
        response = generate_response(msg)
        
        # Reply with the generated message
        line_bot_api.reply_message(tk, TextSendMessage(text=response))
    except Exception as e:
        logging.error(f"Error: {e}")
        logging.error(f"Body: {body}")
    
    return 'OK'

if __name__ == "__main__":
    app.run()
