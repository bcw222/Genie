import sys
import os

os.environ['HUBERT_MODEL_PATH'] = os.getcwd() + "/models/chinese-hubert-base.onnx"
os.environ['OPEN_JTALK_DICT_DIR'] = os.getcwd() + "/models/open_jtalk_dic_utf_8-1.11"\

try:
    import genie_tts as genie
    print("genie_tts imported successfully.")
except ImportError:
    print("Error: genie_tts not found. Please ensure it's installed.")
    sys.exit(1)

try:
    print("Starting FastAPI server...")
    genie.start_server(
        host="0.0.0.0",
        port=8000,
        workers=1
    )
except Exception as e:
    print(f"Error starting FastAPI server: {e}")
    sys.exit(1)