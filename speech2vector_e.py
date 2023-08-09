import speech_recognition as sr
from transformers import BertTokenizer, BertModel
import torch

def recognize_speech(speech):
    with sr.Microphone() as mic:
        print("Started recognizing...")
        audio = speech.listen(mic, phrase_time_limit=10)  # Listen for 10 seconds
        print('...')
    
    try:
        text = speech.recognize_google(audio, language='vi-VN')
    except sr.UnknownValueError:
        text = "Couldn't understand the audio."
    except sr.RequestError as e:
        text = f"Error: {e}"
    
    return text

def embed_text_with_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    speech = sr.Recognizer()

    print('You can start speaking. Say "over" to end the program.')

    while True:
        text = recognize_speech(speech)
        print('Result:', text)

        if 'over' in text:
            break

        embedding = embed_text_with_bert(text, tokenizer, model)
        print(embedding)

if __name__ == "__main__":
    main()
