import librosa
import numpy as np
import os
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DATASET_PATH = "Amharic Voice commands/Class_40"
JSON_PATH = "datasets/Mel_Spec_dataset.json"
SAMPLES_TO_CONSIDER = 32000  # 2sec. worth of sound when loaded in librosa


def prepare_dataset(dataset_path, json_path, n_fft=480, hop_length=240, window='hann', n_mels=20):

    # data dictionary
    data = {
        "Transcription": [],
        "Labels": [],
        "log_mel_spectrogram": [],
        "Files": []
    }

    # loop through all the sub-directories
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # we need to ensure we're not at root level
        if dirpath is not dataset_path:

            # update Mappings
            catagory = dirpath.split("\\")[-1]
            data["Transcription"].append(catagory)
            print(f"Processing {catagory}")

            # loop through all the filenames and extract the mfccs
            for f in filenames:

                # Get the filepath
                file_path = os.path.join(dirpath, f)

                # load the audio file
                signal, sr = librosa.load(file_path, sr=16000)

                # ensure the audio file is at least 1sec.
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # enforce 1sec. long signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract the Melspectrogram
                    Mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft,
                                                              hop_length=hop_length,
                                                              window=window,
                                                              n_mels=n_mels)
                    log_mel_spectrogram = librosa.amplitude_to_db(Mel_spec)

                    # store the data
                    data["Labels"].append(i-1)  # subtracting 1 from the index of the current directory
                    data["log_mel_spectrogram"].append(log_mel_spectrogram.T.tolist()) ### to store the transpose of log_mel_spectrogram (represent n_mles in x-axis and timestamps/frames in y-axis
                    data["Files"].append(file_path)
                    print(f"{file_path}, {i-1}")
                    print(" log_mel_spectrogram Shape")
                    print(log_mel_spectrogram.shape)

                ### Adding Silence to the end of the audio file
                if len(signal) < SAMPLES_TO_CONSIDER:
                    # Calculate the number of samples to add as silence
                    silence_samples = SAMPLES_TO_CONSIDER - len(signal)

                    # Generate an array of zeros (silence) with the calculated number of samples
                    silence = np.zeros(silence_samples)

                    # Concatenate the silence to the end of the signal
                    signal = np.concatenate([signal, silence])

                    # extract the Melspectrogram
                    Mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft,
                                                              hop_length=hop_length,
                                                              window=window,
                                                              n_mels=n_mels)
                    log_mel_spectrogram = librosa.amplitude_to_db(Mel_spec)

                    # store the data
                    data["Labels"].append(i - 1)  # subtracting 1 from the index of the current directory
                    data["log_mel_spectrogram"].append(
                        log_mel_spectrogram.T.tolist())  ### to store the transpose of log_mel_spectrogram (represent n_mles in x-axis and timestamps/frames in y-axis
                    data["Files"].append(file_path)
                    print(f"{file_path}, {i - 1}")
                    print(" log_mel_spectrogram Shape")
                    print(log_mel_spectrogram.shape)
                    ### the shape will be (#mel-bands, #frames)
                    ### (#frames = (total samples/hop-size) +1 -> for 1sec audio)
                    ### (#frames = ((total samples/hop-size)*2) +1 -> for 2secs audio)
                    ### (hop-size = frame-size/2 = fft/2 = 480/2 = 240)
                    ### (#frames = (16,000/240) +1 = 67 -> for 1sec audio)
                    ### (#frames = ((16,000/200)*2) +1 = 134 -> for 2sec audio)
                    ### (#mel-bands = 128)

    # sore data in JSON
    with open(json_path, "w", encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)