import warnings
import wave
import librosa
import numpy as np
import pyaudio
import tensorflow.keras as keras


# the file name output you want to record into
Recorded_file = "recorded.wav"
# set the chunk size of 1024 samples
chunk = 1024
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 44100 samples per second
sample_rate = 22050
record_seconds = 3
# initialize PyAudio object
p = pyaudio.PyAudio()
# open stream object as input & output
stream = p.open(format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                output=True,
                frames_per_buffer=chunk)
frames = []
print("Recording...")
for i in range(int(sample_rate / chunk * record_seconds)):
    data = stream.read(chunk)
    # if you want to hear your voice while recording
    # stream.write(data)
    frames.append(data)
print("Finished recording.")
# stop and close stream
stream.stop_stream()
stream.close()
# terminate pyaudio object
p.terminate()
# save audio file
# open the file in 'write bytes' mode
wf = wave.open(Recorded_file, "wb")
# set the channels
wf.setnchannels(channels)
# set the sample format
wf.setsampwidth(p.get_sample_size(FORMAT))
# set the sample rate
wf.setframerate(sample_rate)
# write the frames as bytes
wf.writeframes(b"".join(frames))
# close the file
wf.close()



MODEL_PATH = "models/CNN_MFCCs_new.h5"
SAMPLES_TO_CONSIDER = 32000  # 2sec. worth of sound when loaded in librosa


class _keyword_spotting_service:
    model = None
    mappings = [
        "ሁሉንም ምረጥ",
        "ለጥፍ",
        "ሙዚቃ ክፈት",
        "ሙዚቃ ዝጋ",
        "ምረጥ",
        "ቀልብስ",
        "ቀዳሚ",
        "ቀጣይ",
        "ቁረጥ",
        "ቅንብሩን ክፈት",
        "ቅዳ",
        "ቆልፍ",
        "በመጠን ደርድር",
        "በስም ደርድር",
        "በቀን ደርድር",
        "በትልቅ አዶ አሳይ",
        "ኖትፓድ ክፈት",
        "ኖትፓድ ዝጋ",
        "አሰይፍ",
        "አሳንስ",
        "አሳድግ",
        "አስምርበት",
        "አስቀምጥ",
        "አትም",
        "አዲስ ክፈት",
        "አድምቅ",
        "አድስ",
        "ካሜራ ክፈት",
        "ካሜራ ዝጋ",
        "ወደቀኝ ተጓዝ",
        "ዝጋ",
        "ዩቱብ ክፈት",
        "ደግመህ ስራ",
        "ድምቀት ቀንስ",
        "ድምቀት ጨምር",
        "ድምጽ ቀንስ",
        "ድምጽ ጨምር",
        "ጎግል ክፈት",
        "ፎቶ ክፈት",
        "ፎቶ ዝጋ"
    ]
    _instance = None

    def predict(self, file_path):
        # extract the MFCCs
        MFCC = self.preprocess(file_path)  # (#segments, #cofficient)

        # convert 2d MFCCs array into 4d MFCCs array -> (#samples, #segments, #cofficients, #channels)
        MFCC = MFCC[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCC)  # [[0.1, 0.6, 0.1.,....]]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self.mappings[predicted_index]

        print(f"The command Predicted index is: {predicted_index}")

        if (predicted_index == 0):
            subprocess.Popen('C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.exe')

        elif (predicted_index == 1):
            """commands to be executed"""
        elif (predicted_index == 2):
            webbrowser.open('https://www.youtube.com/')

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, hop_length=240, n_fft=480, window="hann"):
        # load the audio file
        signal, sr = librosa.load(file_path, sr=16000)

        # ensure consistency in the audio file length
        if len(signal) > SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]

        ### Adding Silence to the end of the audio file
        else:
            # Calculate the number of samples to add as silence
            silence_samples = SAMPLES_TO_CONSIDER - len(signal)

            # Generate an array of zeros (silence) with the calculated number of samples
            silence = np.zeros(silence_samples)

            # Concatenate the silence to the end of the signal
            signal = np.concatenate([signal, silence])

        # extract the MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                     hop_length=hop_length, window=window)

        return MFCCs.T


def keyword_spotting_service():
    # ensure that  we only have 1 instance of KSS
    if _keyword_spotting_service._instance is None:
        _keyword_spotting_service._instance = _keyword_spotting_service()
        _keyword_spotting_service.model = keras.models.load_model(MODEL_PATH)

    return _keyword_spotting_service._instance


if __name__ == "__main__":
    kss = keyword_spotting_service()
    filename = "recorded.wav"
    keyword1 = kss.predict(filename)
    # keyword2 = kss.predict("Class_27/አድስ/refresh (105).wav")
    # keyword3 = kss.predict("Class_27/በቀኝ ጠቅ አድርግ/right_click (27).wav")
    print(f"Predicted keywords: {keyword1} ")

