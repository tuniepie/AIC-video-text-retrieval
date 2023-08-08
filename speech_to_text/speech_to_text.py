import speech_recognition as sr

def speech_to_text(speech):
  print("Started recognizing...")
  while True:
    with sr.Microphone() as mic:
        audio = speech.listen(mic, phrase_time_limit=10)  # Listen in 10 seconds
        print('...')

    try:
        text = speech.recognize_google(audio, language='vi-VN')
    except sr.UnknownValueError:
        text = "Couldn't understand the audio."
    except sr.RequestError as e:
        text = f"Error: {e}"

    print('Result:', text)

    if 'over' in text:
        break

def main():
  speech = sr.Recognizer()
  print('You can start speaking. Say "over" to end the program.')
  speech_to_text(speech)

if __name__ == "__main__":
    main()