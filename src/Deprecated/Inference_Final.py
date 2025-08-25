import time
import onnxruntime
import numpy as np
import pyaudio
import librosa

# 从您指定的模块中导入新的 NumPy 音频处理函数
from Source.Modules.TTS.GSV_Local.Audio_np import preprocess_audio_for_hubert_np, resample_np
from Source.Modules.TTS.GSV_Local.JapaneseToPhones import japanese_to_phones

# 设置 ONNX Runtime 的日志级别为错误级，以减少不必要的输出
onnxruntime.set_default_logger_severity(3)


# ================================================================
# 1. 配置区 (Configuration)
# ================================================================
class Config:
    CHARACTER = 'irene'

    # --- 设备配置 (使用 onnxruntime 检测 CUDA) ---
    DEVICE = "cuda" if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else "cpu"
    SUFFIX = '_fp32' if DEVICE == 'cpu' else '_fp16'
    GENERATE_PERFORMANCE_REPORT = False

    # --- 路径配置 ---
    PATH_PREFIX = "C:/Users/Haruka/Desktop/Midori/Dockers"
    HUBERT_MODEL_PATH = f"{PATH_PREFIX}/mnt/Midori/Onnx_Model/chinese-hubert-base.onnx"
    ONNX_MODEL_DIR = r"C:\Users\Haruka\Desktop\Midori\Dockers\mnt\Midori\GPT_SoVITS\onnx\irene"
    REF_AUDIO_PATH = f"{PATH_PREFIX}/mnt/Midori/WAV/irene_wav/2616341.wav"
    OUTPUT_WAV_PATH = f"C:/Users/Haruka/Desktop/jp_final.wav"
    T2S_ENCODER_PATH = f"{ONNX_MODEL_DIR}/t2s_encoder.onnx"
    T2S_FIRST_STAGE_DECODER_PATH = f"{ONNX_MODEL_DIR}/t2s_first_stage_decoder{SUFFIX}.onnx"
    T2S_STAGE_DECODER_PATH = f"{ONNX_MODEL_DIR}/t2s_stage_decoder{SUFFIX}.onnx"
    VOCODER_PATH = f"{ONNX_MODEL_DIR}/vits.onnx"

    # --- 推理文本配置 (日语) ---
    REF_TEXT_JP = "殺し合いなどせずとも、争いを解決できる方法はきっとあります。"
    SYNTHESIS_TEXT_JP = "こんにちは、世界。"

    # --- T2S解码器参数 ---
    HZ = 50
    MAX_SEC = 54
    EOS_TOKEN = 1024
    BERT_FEATURE_DIM = 1024


config = Config()


# ================================================================
# 2. 核心功能函数 (Core Functions)
# ================================================================


# ================================================================
# 3. 主执行流程 (Main Execution)
# ================================================================
def main():
    print(f"使用的推理设备: {config.DEVICE}")

    # 恢复 providers 的动态设置
    if config.DEVICE == "cuda":
        cuda_provider = ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        })
        providers = [cuda_provider, "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    print("开始加载模型...")
    models = {
        'ssl': load_onnx_model(config.HUBERT_MODEL_PATH, providers),
        't2s': [
            load_onnx_model(config.T2S_ENCODER_PATH, providers),
            load_onnx_model(config.T2S_FIRST_STAGE_DECODER_PATH, providers),
            load_onnx_model(config.T2S_STAGE_DECODER_PATH, providers,
                            generate_report=config.GENERATE_PERFORMANCE_REPORT),
        ],
        'vocoder': load_onnx_model(config.VOCODER_PATH, providers)
    }
    print("所有模型加载完毕。")

    while True:
        synthesis_text_jp = input("输入日文 (直接回车使用默认文本): ")
        if not synthesis_text_jp:
            synthesis_text_jp = config.SYNTHESIS_TEXT_JP

        # --- 日语数据准备 (Japanese Data Preparation) ---
        print("开始准备日语输入数据...")
        timer.start()

        # 1. 将日语文本转换为音素列表 (使用 NumPy)
        ref_phonemes = japanese_to_phones(config.REF_TEXT_JP)
        synthesis_phonemes = japanese_to_phones(synthesis_text_jp)
        ref_seq = np.array([ref_phonemes], dtype=np.int64)
        text_seq = np.array([synthesis_phonemes], dtype=np.int64)
        ref_bert = np.zeros((len(ref_phonemes), config.BERT_FEATURE_DIM), dtype=np.float32)
        text_bert = np.zeros((len(synthesis_phonemes), config.BERT_FEATURE_DIM), dtype=np.float32)
        timer.check('成功将日语文本转换为音素列表')

        # 2. 加载并处理参考音频 (使用 NumPy 函数)
        ref_audio_16k_np = preprocess_audio_for_hubert_np(config.REF_AUDIO_PATH, target_sampling_rate=16000)
        ssl_content = models['ssl'].run(None, {'input_values': ref_audio_16k_np})[0]

        wav_orig, sr_orig = librosa.load(config.REF_AUDIO_PATH, sr=None)
        ref_audio_32k_np = resample_np(wav_orig, orig_sr=sr_orig, new_sr=32000)
        ref_audio_32k_np = np.expand_dims(ref_audio_32k_np, axis=0)
        timer.check('成功加载并处理参考音频')

        # 3. T2S (恢复设备判断逻辑)
        print("文本到语义 (T2S)...")
        if config.DEVICE == 'cuda':
            pred_semantic = t2s_io_binding_final(
                ref_seq, text_seq, ref_bert, text_bert, ssl_content, models
            )
        else:
            pred_semantic = t2s_cpu(
                ref_seq, text_seq, ref_bert, text_bert, ssl_content, models
            )
        timer.check('T2S')

        # 4. S2A (直接传递Numpy数组)
        print("语义到波形 (Vocoder)...")
        audio_output = models['vocoder'].run(None, {
            "text_seq": text_seq,
            "pred_semantic": pred_semantic,
            "ref_audio": ref_audio_32k_np
        })[0]
        timer.check('S2A')
        timer.end()

        # 5. 播放生成的音频
        audio = audio_output.squeeze()
        audio = (audio * 32767).astype(np.int16)

        print("正在播放音频...")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=32000,
                        output=True)
        stream.write(audio.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("播放完毕。\n")


if __name__ == "__main__":
    main()
