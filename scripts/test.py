import wave
import numpy as np
import sys

def analyze_wav(file_path):
    # 打开 wav 文件
    with wave.open(file_path, 'rb') as wf:
        channels = wf.getnchannels()          # 声道数
        sample_width = wf.getsampwidth()      # 采样宽度（字节）
        frame_rate = wf.getframerate()        # 采样率
        n_frames = wf.getnframes()            # 总帧数
        duration = n_frames / frame_rate      # 时长（秒）

        print("🎵 WAV 文件信息:")
        print(f"文件路径: {file_path}")
        print(f"声道数: {channels}")
        print(f"采样宽度: {sample_width} 字节")
        print(f"采样率: {frame_rate} Hz")
        print(f"总帧数: {n_frames}")
        print(f"时长: {duration:.2f} 秒")

        # 读取所有帧
        frames = wf.readframes(n_frames)
        # 转 numpy 数组（小端字节序）
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        audio_data = np.frombuffer(frames, dtype=dtype_map[sample_width])

        # 如果是立体声，取平均
        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)
            audio_data = audio_data.mean(axis=1)

        print("\n📊 波形统计:")
        print(f"最大值: {np.max(audio_data)}")
        print(f"最小值: {np.min(audio_data)}")
        print(f"平均值: {np.mean(audio_data):.2f}")

        # 简单估算主频率（快速傅里叶变换）
        fft_spectrum = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(fft_spectrum), d=1/frame_rate)
        magnitude = np.abs(fft_spectrum)

        # 取正频率部分
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        magnitude = magnitude[pos_mask]

        # 找到能量最大的频率
        dominant_freq = freqs[np.argmax(magnitude)]
        print(f"\n🔑 主频率（估算）: {dominant_freq:.2f} Hz")

if __name__ == "__main__":
    file_path = r'C:\Users\Administrator\Desktop\recording_1756816298757.wav'
    analyze_wav(file_path=file_path)
