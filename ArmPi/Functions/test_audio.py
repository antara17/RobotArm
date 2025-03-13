import sys
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

sys.path.append('/home/pi/ArmPi/')


# Load the Vosk model
model = Model("vosk-model-small-en-us-0.15")
# Create a queue to hold audio data
q = queue.Queue()

# Callback function to collect audio data
# def audio_callback(indata, frames, time, status):
#     if status:
#         print(status, flush=True)
#     q.put(bytes(indata))

# Query default input device and get its samplerate
try:
    device_info = sd.query_devices(None, 'input')
    samplerate = int(device_info['default_samplerate'])
except Exception as e:
    print("Error querying device: ", e)
    sys.exit(1)

# Initialize recognizer with the appropriate samplerate
recognizer = KaldiRecognizer(model, samplerate)

# Open an input audio stream
with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None, dtype='int16',
                        channels=1) as stream:
    print("Listening...")
    while True:
        data, _ = stream.read(4000)
        data_bytes = data.tobytes()  # convert to bytes
        if recognizer.AcceptWaveform(data_bytes):
            result = json.loads(recognizer.Result())
            print(result)
            if "time to go eat" in result.get("text", "").lower():
                print("found the correct phrase")