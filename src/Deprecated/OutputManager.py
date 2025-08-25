# Source/Core/output_manager.py

import queue
import threading
import logging
import numpy as np
import simpleaudio as sa
import soundfile as sf
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# 定义音频任务的数据结构
@dataclass
class AudioTask:
    audio_bytes: np.ndarray  # TTS生成的原始浮点数numpy数组
    play: bool = False  # 是否直接播放
    save_path: Optional[str] = None  # 如果需要保存，指定文件路径
    stream_queue: Optional[queue.Queue] = None  # 用于FastAPI流式传输的特定队列


class OutputManager:
    """
    管理TTS生成的音频数据，并将其分发到不同的输出目标：
    1. 直接播放
    2. 写入文件
    3. 用于FastAPI流式传输
    """

    def __init__(self, sample_rate: int = 32000):
        self.SAMPLE_RATE = sample_rate
        self._task_queue: queue.Queue[AudioTask] = queue.Queue()
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True, name="AudioOutputWorker")
        self.worker_thread.start()

    @staticmethod
    def _preprocess_for_playback(audio_float: np.ndarray) -> bytes:
        """将浮点numpy数组转换为16-bit PCM字节流以供播放。"""
        # 确保数组是一维的
        audio_squeezed = audio_float.squeeze()
        # 将浮点数范围 (-1.0, 1.0) 转换为 16-bit 整数范围 (-32768, 32767)
        audio_int16 = (audio_squeezed * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def add_task(self, task: AudioTask) -> None:
        """向队列中添加一个新的音频处理任务。"""
        self._task_queue.put(task)

    def _worker(self):
        """工作线程，持续从队列中获取并处理任务。"""
        while not self._stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                if task.play:
                    self._play_audio(task.audio_bytes)

                if task.save_path:
                    self._write_to_file(task.audio_bytes, task.save_path)

                if task.stream_queue:
                    self._stream_audio(task.audio_bytes, task.stream_queue)

            except Exception as e:
                logger.error(f"处理音频任务时出错: {e}", exc_info=True)
            finally:
                self._task_queue.task_done()

    def _play_audio(self, audio_float: np.ndarray):
        """使用 simpleaudio 播放音频，并等待其播放完成。"""
        try:
            logger.info("准备播放音频...")
            audio_data = self._preprocess_for_playback(audio_float)

            play_obj = sa.play_buffer(
                audio_data,
                num_channels=1,
                bytes_per_sample=2,
                sample_rate=self.SAMPLE_RATE
            )
            logger.info("音频正在播放... 将等待播放完成。")

            # 关键改动：在这里阻塞，直到当前音频播放结束
            play_obj.wait_done()

            logger.info("音频播放完毕。")
        except Exception as e:
            logger.error(f"播放音频失败: {e}", exc_info=True)

    def _write_to_file(self, audio_float: np.ndarray, path: str):
        """使用 soundfile 将音频写入WAV文件。"""
        try:
            logger.info(f"正在将音频写入文件: {path}")
            # soundfile可以直接处理float类型的numpy数组，非常方便
            sf.write(path, audio_float.squeeze(), self.SAMPLE_RATE)
            logger.info(f"音频已成功保存到: {path}")
        except Exception as e:
            logger.error(f"写入音频文件失败: {e}", exc_info=True)

    def _stream_audio(self, audio_float: np.ndarray, stream_q: queue.Queue):
        """将处理后的音频字节块放入指定的流队列中。"""
        try:
            logger.info("正在将音频块添加到流队列...")
            # 为了流式传输，我们也发送16-bit PCM字节流
            audio_data = self._preprocess_for_playback(audio_float)
            stream_q.put(audio_data)
        except Exception as e:
            logger.error(f"流式传输音频块失败: {e}", exc_info=True)

    def stop(self):
        """停止工作线程并清空队列。"""
        logger.info("正在停止 OutputManager...")
        self._stop_event.set()
        # 清空队列，防止线程在退出后仍在等待
        with self._task_queue.mutex:
            self._task_queue.queue.clear()
        self.worker_thread.join()  # 等待线程完全终止
        logger.info("OutputManager 已停止。")


# 创建一个全局单例
output_manager = OutputManager()
