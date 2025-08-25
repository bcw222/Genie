import queue
import re
import logging
import threading
import time
from typing import Optional
import numpy as np

from Source.Core.Inference import tts_client
from Source.Utils.Shared import context
from Source.ModelManager import model_manager, GSVModel

logger = logging.getLogger(__name__)

MIN_SENTENCE_LENGTH = 7

# 定义用于分割句子的标点
SENTENCE_TERMINATORS = "。！？…"
# 定义有效字符的正则表达式，用于精确计算长度
VALID_CHAR_PATTERN = re.compile(
    r'[\u3040-\u309F'  # 平假名
    r'\u30A0-\u30FF'  # 片假名
    r'\u4E00-\u9FFF'  # 汉字
    r'a-zA-Z'  # 半角字母
    r'\uFF21-\uFF3A\uFF41-\uFF5A'  # 全角字母
    r'0-9'  # 半角数字
    r'\uFF10-\uFF19'  # 全角数字
    r']'
)


def _get_valid_text_length(sentence: str) -> int:
    return len(VALID_CHAR_PATTERN.findall(sentence))


def _split_japanese_text(long_text: str) -> list[str]:
    if not long_text:
        return []
    # 使用正向后行断言 `(?<=...)` 来在分割时保留分隔符
    raw_sentences = re.split(f'(?<=[{SENTENCE_TERMINATORS}])', long_text)
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not raw_sentences:
        return [long_text] if long_text.strip() else []

    final_sentences = []
    for sentence in raw_sentences:
        clean_len = _get_valid_text_length(sentence)
        # 如果final_sentences不为空，且上一句的有效长度也小于最小长度，或者当前句本身就小于最小长度，则合并
        if final_sentences and clean_len < MIN_SENTENCE_LENGTH:
            final_sentences[-1] += sentence
        else:
            final_sentences.append(sentence)
    return final_sentences


class InputManager:
    def __init__(self):
        self._task_queue: queue.Queue[str] = queue.Queue()
        self._stream_buffer: str = ""
        self.worker_thread: Optional[threading.Thread] = threading.Thread(target=self._worker, daemon=True,
                                                                          name="TTSTaskWorker")
        self.worker_thread.start()

    def add_text_to_queue(self, long_japanese_text: str) -> None:
        sentences = _split_japanese_text(long_japanese_text.strip())
        for sentence in sentences:
            self._task_queue.put(sentence)

    def stream_add(self, chunk: str) -> None:
        if not isinstance(chunk, str):
            logger.error("流式输入只接受字符串类型。")
            return
        self._stream_buffer += chunk
        last_terminator_pos = -1
        for term in SENTENCE_TERMINATORS:
            last_terminator_pos = max(last_terminator_pos, self._stream_buffer.rfind(term))

        if last_terminator_pos != -1:
            text_to_process = self._stream_buffer[:last_terminator_pos + 1]
            self._stream_buffer = self._stream_buffer[last_terminator_pos + 1:]
            self.add_text_to_queue(text_to_process)

    def stream_end(self) -> None:
        if self._stream_buffer.strip():
            self.add_text_to_queue(self._stream_buffer)
        self._stream_buffer = ""  # 清空缓冲区

    def stop(self) -> None:  # 阻塞，直到确实完全停止。
        tts_client.stop_event.set()
        # 清空队列中所有待处理的任务。
        with self._task_queue.mutex:
            self._task_queue.queue.clear()
        # 清空流式输入的缓冲区
        self._stream_buffer = ""
        self._task_queue.join()
        tts_client.stop_event.clear()

    def _worker(self):
        while True:
            try:
                text: str = self._task_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                start_time = time.time()

                if not context.current_prompt_audio or not context.current_speaker:
                    continue
                gsv_model: Optional[GSVModel] = model_manager.get(context.current_speaker)
                if not gsv_model:
                    continue
                audio_bytes: Optional[np.ndarray] = tts_client.tts(
                    text=text,
                    prompt_audio=context.current_prompt_audio,
                    encoder=gsv_model.T2S_ENCODER,
                    first_stage_decoder=gsv_model.T2S_FIRST_STAGE_DECODER,
                    stage_decoder=gsv_model.T2S_STAGE_DECODER,
                    vocoder=gsv_model.VITS,
                )
                if not audio_bytes:
                    continue

                duration = time.time() - start_time
                logging.info(f"推理完成，耗时: {duration:.2f}s")
            except Exception as e:
                logger.error(f"TTS 时发生意外错误: {e}", exc_info=True)
            finally:
                self._task_queue.task_done()


input_manager: InputManager = InputManager()
