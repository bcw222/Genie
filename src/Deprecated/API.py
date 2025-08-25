from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import logging
from Source.Core.TTSPlayer import tts_player

logger: logging.Logger = logging.getLogger(__name__)
app: FastAPI = FastAPI()


async def audio_stream_generator(text_to_process: str, save_to_file: bool):
    stream_q = asyncio.Queue()
    save_path = "fastapi_output.wav" if save_to_file else None

    # 1. 启动一个包含所有输出目标的会话
    tts_player.start_session(save_path=save_path, stream_queue=stream_q)

    # 2. 提交任务
    tts_player.feed(text_to_process)
    tts_player.end_session()

    # 3. 从队列中消费并 yield 给客户端
    try:
        while True:
            chunk = await stream_q.get()
            if chunk is None:
                break
            yield chunk
            stream_q.task_done()
    except Exception as e:
        logger.error(f"音频流生成时出错: {e}")
    finally:
        logger.info("音频流生成器结束。")


@app.post("/tts/speak")
async def speak_and_stream(text: str, save: bool = False):
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    generator = audio_stream_generator(text_to_process=text, save_to_file=save)
    return StreamingResponse(generator, media_type="audio/wav")
