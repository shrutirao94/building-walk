import whisper

PATH_TO_DATA = '/content/drive/MyDrive/UvA-DIL/user-study-1/building-walk/data/'
files = ['230606_133630_00.mp3', '230607_134321_00.mp3', '230608_140602_00.mp3']

model = whisper.load_model("base")

for file in files:
  print(PATH_TO_DATA + file)
  result = model.transcribe(PATH_TO_DATA + file)

  with open(file + '.txt', 'w') as f:
    f.write(result['text'])
    f.write('\n')


