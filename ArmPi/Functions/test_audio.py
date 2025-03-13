import sys
import json
import threading
import sounddevice as sd
from vosk import Model, KaldiRecognizer

sys.path.append('/home/pi/ArmPi/')

# Load the Vosk model
model = Model("vosk-model-small-en-us-0.15")

# Query default input device and get its samplerate
try:
    device_info = sd.query_devices(None, 'input')
    samplerate = int(device_info['default_samplerate'])
except Exception as e:
    print("Error querying device:", e)
    sys.exit(1)

# Initialize recognizer with the appropriate samplerate
recognizer = KaldiRecognizer(model, samplerate)

def audio_thread():
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None, dtype='int16',
                           channels=1) as stream:
        print("Listening...")
        while True:
            # Read a chunk of audio data directly from the stream
            data, _ = stream.read(4000)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                print("Final result:", result)
                # Check for a specific trigger phrase in the recognized text
                if "time for downward dog" in result.get("text", "").lower():
                    print("Trigger phrase detected!")
                    # Call your trigger function here
            # else:
            #     partial_result = json.loads(recognizer.PartialResult())
            #     print("Partial result:", partial_result)

# Create and start the audio processing thread
thread = threading.Thread(target=audio_thread, daemon=True)
thread.start()

# Continue with other code in the main thread as needed
while True:
    pass  # Replace with your main thread tasks or a proper exit condition