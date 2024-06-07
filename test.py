from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from token_and_secret import token, secret
from chat import *

import json
import logging

app = Flask(__name__)

# Setup logger
logging.basicConfig(level=logging.INFO)

# Initialize LineBotApi and WebhookHandler with your token and secret
line_bot_api = LineBotApi(token)
handler = WebhookHandler(secret)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)  # Get the request body as text
    logging.info(f"Request body: {body}")
    
    signature = request.headers.get('X-Line-Signature', '')  # Get X-Line-Signature header value
    logging.info(f"Signature: {signature}")
    
    try:
        # Handle webhook body and signature
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)  # If signature is invalid, return 400 Bad Request
    
    try:
        # Parse the request body to JSON
        json_data = json.loads(body)
        logging.info(f"JSON data: {json_data}")
        
        # Get the message text and reply token
        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        
        print(chat_fun(msg))

        # Reply with the same message
        line_bot_api.reply_message(tk, TextSendMessage(text=msg))
    except Exception as e:
        logging.error(f"Error: {e}")
        logging.error(f"Body: {body}")
    
    return 'OK'

if __name__ == "__main__":
    app.run()
