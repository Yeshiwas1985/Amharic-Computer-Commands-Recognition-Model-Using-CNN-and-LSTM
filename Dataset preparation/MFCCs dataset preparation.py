import librosa
import numpy as np
import os
import json
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

DATASET_PATH = "Amharic Voice commands/Class_40"
JSON_PATH = "datasets/MFCCs_dataset1.json"
SAMPLES_TO_CONSIDER = 32000  # 2sec. worth of sound when loaded in librosa


def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=240, n_fft=480):
    # data dictionary
    data = {
        "Transcription": [],
        "Labels": [],
        "MFCCs": [],
        "Files": []
    }

    # loop through all the sub-directories
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # we need to ensure we're not at root level
        if dirpath is not dataset_path:

            # update Mappings
            category = dirpath.split("\\")[-1]
            data["Transcription"].append(category)
            print(f"Processing {category}")

            # loop through all the filenames and extract the MFCCs
            for f in filenames:

                # Get the filepath
                file_path = os.path.join(dirpath, f)

                # load the audio file
                signal, sr = librosa.load(file_path, sr=16000)

                # ensure the audio file is at least 1sec.
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # enforce 2sec. long signal, considering only the 2.0 sec duration of the signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract the mfcc
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                    # store the data
                    data["Labels"].append(i - 1)  # subtracting 1 from the index of the current directory

                    # storing the transpose of MFCCs (to store MFCCs coefficients in x-axis and timestamps/frames in y-axis)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["Files"].append(file_path)
                    print(f"{file_path}, {i - 1}")
                    print(" MFCCs Shape: ", MFCCs.shape)

                ### Adding Silence to the end of the audio file
                if len(signal) < SAMPLES_TO_CONSIDER:
                    # Calculate the number of samples to add as silence
                    silence_samples = SAMPLES_TO_CONSIDER - len(signal)

                    # Generate an array of zeros (silence) with the calculated number of samples
                    silence = np.zeros(silence_samples)

                    # Concatenate the silence to the end of the signal
                    signal = np.concatenate([signal, silence])

                    # extract the mfcc
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                    # store the data
                    data["Labels"].append(i - 1)  # subtracting 1 from the index of the current directory

                    # storing the transpose of MFCCs (to store MFCCs coefficients in x-axis and timestamps/frames in y-axis)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["Files"].append(file_path)
                    print(f"{file_path}, {i - 1}")
                    print(" MFCCs Shape: ", MFCCs.shape)

                    ### the shape will be (#coefficients, #frames)
                    ### (#frames = (total samples/hop-size) +1 -> for 1sec audio)
                    ### (#frames = ((total samples/hop-size)*2) +1 -> for 2secs audio)
                    ### (hop-size = frame-size/2 = fft/2 = 480/2 = 240)
                    ### (#frames = (16,000/240) +1 = 67 -> for 1sec audio)
                    ### (#frames = ((16,000/240)*2) +1 = 134 -> for 2sec audio))
                    ### (#coefficients = 13)
                    ### print(f"length of signal: {1/sr * len(signal):2f} seconds")
                    ### print(f"Duration of 1 frame:{n_fft/(2*sr): 6f} seconds")
                    ### print(f"Frame step is:{hop_length/(2*sr): 6f} seconds")

    # sore data in JSON
    with open(json_path, "w", encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)
