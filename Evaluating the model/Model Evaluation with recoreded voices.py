# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 22:21:21 2022

@author: Yeshiwas
"""

import tensorflow.keras as keras
import numpy as np
import librosa
import webbrowser
import subprocess
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
        predicted_keyword  = self.mappings[predicted_index]

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

    keyword1 = kss.predict("test_data/አድስ2.wav")
    keyword2 = kss.predict("test_data/ምረጥ 3.wav")
    keyword3 = kss.predict("test_data/ድምፅ ቀንስ.wav")
    # keyword4 = kss.predict("test_data/Volume_minus.wav")
    # keyword5 = kss.predict("test_data/Br_max.wav")
    # keyword6 = kss.predict("test_data/Br_min.wav")
    # keyword7 = kss.predict("test_data/Close.wav")
    # keyword8 = kss.predict("test_data/Size_order.wav")
    # keyword9 = kss.predict("test_data/Italic(2).wav")
    # keyword10 = kss.predict("test_data/Next.wav")
    # keyword11 = kss.predict("test_data/New.wav")
    # keyword12 = kss.predict("test_data/Underline.wav")
    # keyword13 = kss.predict("test_data/Save.wav")
    # keyword14 = kss.predict("test_data/ምረጥ.wav")
    # keyword15 = kss.predict("test_data/አድስ.wav")
    # keyword16 = kss.predict("test_data/Undo.wav")
    # keyword17 = kss.predict("test_data/Redo.wav")
    # keyword18 = kss.predict("test_data/Pho_op.wav")
    # keyword19 = kss.predict("test_data/Pho_cl.wav")
    # keyword20 = kss.predict("test_data/Min.wav")

    
    print(f"Predicted keyword1: {keyword1} ")
    print(f"Predicted keyword2: {keyword2} ")
    print(f"Predicted keyword3: {keyword3} ")
    # print(f"Predicted keyword4: {keyword4} ")
    # print(f"Predicted keyword5: {keyword5} ")
    # print(f"Predicted keyword6: {keyword6} ")
    # print(f"Predicted keyword7: {keyword7} ")
    # print(f"Predicted keyword8: {keyword8} ")
    # print(f"Predicted keyword9: {keyword9} ")
    # print(f"Predicted keyword10: {keyword10} ")
    # print(f"Predicted keyword11: {keyword11} ")
    # print(f"Predicted keyword12: {keyword12} ")
    # print(f"Predicted keyword13: {keyword13} ")
    # print(f"Predicted keyword14: {keyword14} ")
    # print(f"Predicted keyword15: {keyword15} ")
    # print(f"Predicted keyword16: {keyword16} ")
    # print(f"Predicted keyword17: {keyword17} ")
    # print(f"Predicted keyword18: {keyword18} ")
    # print(f"Predicted keyword19: {keyword19} ")
    # print(f"Predicted keyword20: {keyword20} ")
