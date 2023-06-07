from huggingsound import SpeechRecognitionModel
from pydub import AudioSegment

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
audio_path = ["../data/pilot/room1_sample.mov"]
# audio_path = ["../data/pilot/sample.mov"]

transcriptions = model.transcribe(audio_path)
transcriptions = transcriptions[0]["transcription"]

with open("../data/pilot/transcription.txt","w+") as f:
        f.writelines(transcriptions)




