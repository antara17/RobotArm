import speech_recognition as sr

# Global flag to control motion triggering
trigger_motion = False

def listen_for_phrase(target_phrase="activate motion"):
    global trigger_motion
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Voice listener active. Say '" + target_phrase + "' to activate motion.")
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source, phrase_time_limit=5)
                recognized_text = recognizer.recognize_google(audio).lower()
                print("Heard: ", recognized_text)
                if target_phrase.lower() in recognized_text:
                    print("Target phrase detected!")
                    trigger_motion = True
                else:
                    # Optionally, reset the flag or ignore non-target phrases
                    trigger_motion = False
            except Exception as e:
                print("Error:", e)
