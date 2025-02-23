import speech_recognition as sr

recognizer = sr.Recognizer()

# Specify the device index for the desired microphone
microphone = sr.Microphone(device_index=2)

print("Please speak something...")

with microphone as source:
    print("Adjusting for ambient noise...")
    recognizer.adjust_for_ambient_noise(source)
    print("Listening for audio...")
    audio = recognizer.listen(source)

try:
    print("Recognizing speech...")
    text = recognizer.recognize_google(audio)
    print("You said: " + text)
except sr.UnknownValueError:
    print("Sorry, I could not understand that.")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition; {e}")
