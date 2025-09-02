from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from openai import OpenAI
from config import *
import requests
from flask import send_file
import struct
import wave
import miniaudio

app = Flask(__name__)
CORS(app)  # 允许跨域请求

def get_image_data_url(image_file: str, image_format: str) -> str:
    import base64
    """
    Helper function to converts an image file to a data URL string.

    Args:
        image_file (str): The path to the image file.
        image_format (str): The format of the image file.

    Returns:
        str: The data URL of the image.
    """
    try:
        with open(image_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Could not read '{image_file}'.")
        exit()
    return f"data:image/{image_format};base64,{image_data}"

def resize_image(file_path, max_size=(500, 500)):
    from PIL import Image
    import os
    """
    将指定图片缩放到不超过 500x500 分辨率（保持比例）。
    :param file_path: str，图片文件路径
    :param max_size: tuple，最大尺寸 (宽, 高)
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(f"文件不存在: {file_path}", file=sys.stderr)
        return
    
    try:
        with Image.open(file_path) as img:
            # 缩放（保持比例）
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 覆盖保存
            img.save(file_path)
            print(f"已处理: {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"跳过 {file_path}: {e}", file=sys.stderr)

def llm_api(prompt, history=None):
    """调用大模型API"""
    # 构建对话历史
    system_prompt = """
        你是一个暖心的AI，以外孙胡戴立的身份陪外婆聊天。

        关心她吃饭没、腿还疼不、麻将赢了没，

        记得提她做的红烧肉、阳台的花、老邻居的事，

        说话像孙子撒娇，每句不超过20字，如"外婆，晚饭吃了吗？"
    """
    system_prompt = "你是一个名为02的二次元动漫人物，性格活泼可爱，回答要简洁有趣，所有回答请保证在100个字以内"
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # 添加历史对话（只保留最近10轮对话以控制长度）
    if history:
        recent_history = history[-HISTORY_LEN:]  # 最近的对话
        
        # 处理图片（如果有的话）
        for entry in recent_history:
            if entry.get("files_path"):
                for item in entry["files_path"]:
                    if item.endswith(('.jpg', '.png', '.jpeg')):
                        resize_image(item.replace("\\", "/"))
        
        # 构建消息历史
        for entry in recent_history:
            if entry.get('role') in ['user', 'assistant']:
                content = [{
                    "type": "text",
                    "text": entry['content']
                }]
                
                # 添加图片（如果有的话）
                if entry.get("files_path"):
                    for item in entry["files_path"]:
                        if item.endswith(('.jpg', '.png', '.jpeg')):
                            image_url = get_image_data_url(item.replace("\\", "/"), item.split('.')[-1])
                            if image_url:
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                        "detail": "low"
                                    },
                                })
                
                messages.append({
                    "role": entry['role'],
                    "content": content if len(content) > 1 else entry['content']
                })
    
    # 添加当前用户输入
    # messages.append({"role": "user", "content": prompt})
    
    print(f"发送的消息: {messages}", file=sys.stderr)
    
    try:
        endpoint = "https://models.github.ai/inference"
        model_name = "openai/gpt-4o"

        client = OpenAI(
            base_url=endpoint,
            api_key=LLM_API,
        )

        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
        )
        
        result = response.choices[0].message.content
        print(f"模型回复如下！！！\n{result}", file=sys.stderr)
        return result
        
    except Exception as e:
        print(f"API调用失败: {e}", file=sys.stderr)
        return "抱歉，我现在无法回应，请稍后再试。"

@app.route('/chat', methods=['POST'])
def chat():
    """聊天接口"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        message = data.get('message', '')
        history = data.get('history', [])
        enable_voice = data.get('enableVoice', False)
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # 调用大模型API
        response_text = llm_api(message, history)
        
        # 构建响应
        response_data = {
            "response": response_text,
            "success": True,
            "enableVoice": enable_voice
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"处理请求时出错: {e}", file=sys.stderr)
        return jsonify({
            "error": "Internal server error",
            "response": "抱歉，服务器出现了问题。",
            "success": False
        }), 500

def mp3_to_wav(mp3_path, wav_path):
    with open(mp3_path, "rb") as f:
        mp3_data = f.read()
    
    # 解码为 PCM 音频
    decoded = miniaudio.decode(mp3_data)
    # decoded.samples 是 PCM 数据（小端16位整数列表）
    # decoded.sample_rate, decoded.nchannels

    # 2. 写入 WAV 文件
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(decoded.nchannels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(decoded.sample_rate)
        # 将样本数据打包为二进制
        packed_data = struct.pack(f"<{len(decoded.samples)}h", *decoded.samples)
        wf.writeframes(packed_data)

@app.route('/tts', methods=['POST'])
def tts():
    """文本转语音接口"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"].replace("\n", "")

        # 调用外部TTS接口
        url = "https://api.tjit.net/api/ai/audio/speech"
        params = {
            "key": TTS_API,
            "text": text,
            "type": "speech"
        }
        response = requests.get(url, params=params)

        if response.status_code != 200:
            return jsonify({"error": "TTS API 调用失败"}), 500

        # 保存mp3
        mp3_path = "audio.mp3"
        with open(mp3_path, "wb") as f:
            f.write(response.content)

        # 转成 wav
        wav_path = "audio.wav"
        mp3_to_wav(mp3_path, wav_path)

        # 返回 wav 文件
        return send_file(
            wav_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="audio.wav"
        )

    except Exception as e:
        print(f"TTS处理失败: {e}", file=sys.stderr)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route('/', methods=['GET'])
def index():
    """根路径"""
    return jsonify({"message": "Flask LLM Server is running!"})

if __name__ == '__main__':
    print("启动 Flask 服务器...")
    print("请确保已设置正确的 LLM_API 密钥")
    
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"服务器将在以下地址启动:")
    print(f"本地访问: http://localhost:5000")
    print(f"局域网访问: http://{local_ip}:5000")
    print(f"HTTPS访问: https://{local_ip}:5000")
    
    # 方案1：使用自签名证书启动 HTTPS 服务器
    # 注意：需要先生成证书文件
    try:
        # 尝试启动 HTTPS 服务器
        app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')
    except Exception as e:
        print(f"HTTPS启动失败: {e}")
        print("回退到 HTTP 模式")
        # 回退到 HTTP
        app.run(host='0.0.0.0', port=5000, debug=True)