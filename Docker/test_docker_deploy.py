"""
Start the API Server using genie.start_server(host=SERVER_HOST, port=SERVER_PORT, workers=1).

Quick Reference for Genie TTS Server API

1. Load Character Model
Endpoint: POST /load_character
Function: Load a character model into the server.
Request Parameters (JSON):
    - character_name (string): Unique name of the character.
    - onnx_model_dir (string): Path to the model folder on the server.

2. Set Reference Audio
Endpoint: POST /set_reference_audio
Function: Set the audio required for voice cloning for the loaded character.
Request Parameters (JSON):
    - character_name (string): Name of the character to set.
    - audio_path (string): Path to the reference audio file on the server.
    - audio_text (string): Text corresponding to the reference audio.

3. Text-to-Speech (TTS)
Endpoint: POST /tts
Function: Generate speech and return it as an audio/wav stream.
Request Parameters (JSON):
    - character_name (string): Character name to use.
    - text (string): Text to convert to speech.
    - split_sentence (boolean, optional): Whether to auto-split sentences, default is false.
    - save_path (string, optional): Full path to save audio on the server.

4. Unload Character Model
Endpoint: POST /unload_character
Function: Remove a character from server memory to free resources.
Request Parameters (JSON):
    - character_name (string): Character name to unload.

5. Stop All TTS Tasks
Endpoint: POST /stop
Function: Immediately stop all ongoing speech synthesis tasks.
Request Parameters: None.

6. Clear Reference Audio Cache
Endpoint: POST /clear_reference_audio_cache
Function: Clear the loaded reference audio cache on the server.
Request Parameters: None.
"""

import requests
import pyaudio

# --- Configuration ---
# Server address
SERVER_HOST = "192.168.50.246"
SERVER_PORT = 9999
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

BYTES_PER_SAMPLE = 2
CHANNELS = 1
SAMPLE_RATE = 32000

def main_client():
    # 1. Load Character
    print("\n[Client] Step 1: Sending load character request...")
    load_payload = {
        "character_name": "misono_mika",  # Replace with your character name
        "onnx_model_dir": r"./models/misono_mika"  # Replace with the folder containing the ONNX model
    }
    try:
        response = requests.post(f"{BASE_URL}/load_character", json=load_payload)
        response.raise_for_status()
        print(f"[Client] Character loaded successfully: {response.json()['message']}")
    except requests.exceptions.RequestException as e:
        print(f"[Client] Failed to load character: {e}")
        return

    # 2. Set Reference Audio
    print("\n[Client] Step 2: Sending set reference audio request...")
    ref_audio_payload = {
        "character_name": "misono_mika",  # Use the same character name as above
        "audio_path": r"./models/misono_mika/prompt.wav",  # Replace with path to your reference audio file
        "audio_text": "私も昔、これと似たようなの持ってたなぁ…。"  # Replace with the text corresponding to the reference audio
    }
    try:
        response = requests.post(f"{BASE_URL}/set_reference_audio", json=ref_audio_payload)
        response.raise_for_status()
        print(f"[Client] Reference audio set successfully: {response.json()['message']}")
    except requests.exceptions.RequestException as e:
        print(f"[Client] Failed to set reference audio: {e}")
        return

    # 3. Request TTS and play audio stream
    print("\n[Client] Step 3: Requesting TTS and preparing audio stream...")
    tts_payload = {
        "character_name": "misono_mika",  # Use the same character name
        "text": "おはようございます",  # Replace with the text you want to synthesize
        "split_sentence": True
    }

    p = pyaudio.PyAudio()
    stream = None

    try:
        with requests.post(f"{BASE_URL}/tts", json=tts_payload, stream=True) as response:
            response.raise_for_status()
            print("[Client] Connected to audio stream, starting playback...")

            # Iterate over received audio chunks
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    if stream is None:
                        stream = p.open(format=p.get_format_from_width(BYTES_PER_SAMPLE),
                                        channels=CHANNELS,
                                        rate=SAMPLE_RATE,
                                        output=True)
                    stream.write(chunk)

            print("[Client] Audio stream finished.")

    except requests.exceptions.RequestException as e:
        print(f"[Client] TTS request failed: {e}")
    except Exception as e:
        print(f"[Client] Error during playback: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()


if __name__ == "__main__":
    # Run client logic
    main_client()
