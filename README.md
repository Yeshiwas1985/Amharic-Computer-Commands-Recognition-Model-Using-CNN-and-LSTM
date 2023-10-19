# Computer Commands and Contro Model Using Amharic_voice_commands in CNN and LSTM Algorithms
## This repository about building a model that can recognizes computer commands in amharic voice
## The model is build in 40 different commands:
|  Amharic command | English equivalent  | Amharic command  |  English equivalent | Amharic command  |  English equivalent |
|---|---|---|---|---|---|
|  1.	ሁሉንም ምረጥ | Select all  | 15.	በቀን ደርድር	  | Sort-by-date   | 29. ወደቀኝ ተጓዝ  | Move to the right  |  
|  2.	ለጥፍ	 | Paste  | 16.	ኖትፓድ ክፈት  | 	Open notepad  | 30. ዝጋ	 | Close  |
|  3.	ሙዚቃ ክፈት	|  Open a music |  17.	ኖትፓድ ዝጋ  | 	Close notepad  | 31. ዩቱብ ክፈት  | Open youtube  |
|  4.	ሙዚቃ ዝጋ	| Close a music  |18.	አሰይፍ	  | Make it italic   | 32. ደግመህ ስራ  | Redo  |
|  5.	ምረጥ	| Select  |19.	አሳንስ  | Minimize   | 33. ድምቀት ቀንስ   |  Decrease a brightness |
|  6.	ቀልብስ	| Undo   | 20.	አሳድግ  | Maximize  | 34.	ድምቀት ጨምር  | Increase a brightness  |
|  7.	ቀዳሚ	 | Previous  | 21.	አስምርበት | Make it underline  | 35. ድምጽ ቀንስ  | Decrease a volume |
|  8.ቀጣይ	 | Next  | 22.	አስቀምጥ  | Save  | 36. ድምጽ ጨምር  | Increase a volume  |
|  9. ቁረጥ  | Cut |  23.	አትም	 | Print  |  37. ጎግል ክፈት |  Open a google |
|  10. ቅንብሩን ክፈት | Open the setting  |24.	አዲስ ክፈት   | 	Open new  | 38.	ጠቅ አድርግ  | click  |
|  11. ቅዳ  | Copy  | 25.	አድምቅ  | 	Make it bold  | 39. ፎቶ ክፈት  | Open a photo  |
|  12. ቆልፍ | Lock  | 26.	አድስ	  | Refresh  | 40. ፎቶ ዝጋ  | Close a photo  |
| 13. በመጠን ደርድር  | Sort-by-size  | 27.	ካሜራ ክፈት  | 	Open the camera  |   |   |
|  14. በስም ደርድር  |Sort-by-name   | 28.	ካሜራ ዝጋ  | Close the camera  |   |   |

## <b>Data Collection:</b>
To develop a model, the required type of data is audio or voice. The voice data is collected using smartphones via voice recorder applications. The built model is speaker-independent, which means it can recognize speech commands regardless of the speaker. So, to solve acoustic-related problems (such as accents and pronunciations), the voices are collected from 35 different Amharic language speakers (20 males and 15 females in the age range of 21-28) at least five times per command. The speech commands are recorded at a sample rate of 16kHhz with the mono channel and then they are stored in .wav audio file format. Most of the voices are recorded in less-noisy environments (i.e. at home and the office, because the acoustic environment and transduction equipment have a great effect). A total of 7000 voices have been collected (all voices are recorded using mobile phones).
<br>
In this study, a 16kHz sampling rate is used so that each voice has a total of 32,000 samples. The maximum length or duration of the speech commands is 2 seconds (all Speakers have finished below 2 seconds) so that the content speech of the first 2 seconds of the speech is considered

## <b> Feature Extraction: </b>
To feed data to the CNN and LSTm models, two audio features ( <b> MFCCs and Mel-Spectrogram </b>) are extracted from the raw audio files.
### Mel-spectrogram
Mel spectrogram is the combination of Mel-Scale + Spectrogram, Mel scale is the logarithmic transformation of an audio signal’s frequency. The idea behind the Mel scale transformation is that an audio signal with equal distance in the Mel scale is perceived to be equal distance by humans. A spectrogram is a visual representation of the "loudness" or signal strength over time at different frequencies contained in a specific waveform.
### Mel-Frequency Cepstral Coefficients (MFCC)
Cepstrum: The Fourier transform of the logarithm of a speech power spectrum; used to separate vocal tract information from pitch excitation in voiced speech. Mel-frequency Cepstrum is the representation of the power spectrum. The power spectrum is derived by applying a Discrete-Fourier Transform and it describes the intensity or energy of a signal in each frequency bin in the frequency domain. Cepstral is a spectrum-of-a-spectrum. MFCC represents the frequency bands equally in the Mel scale, which mimics the human auditory system, which makes the key audio feature for different audio signal processing tasks.

## <b> Data processing:</b>
The raw audio has continuous values, which is difficult for the machines to extract the features from the specified period. To extract the selected features from the speech signal, it must be divided into equal parts called frames. The following figure shows the raw voice signal that has a continuous value.
<br/>
To get discrete values, the voice signal must be divided into equal segments which have an equal number of points or samples. This process is called framing. It is used to divide a longer-time signal into shorter segments (called frames) with equal length or duration (each frame has an equal number of samples) and used to know how these different frequency components of a signal are available over time. The recommended duration of each frame (window length) is 20ms-40ms.
<br/>
In this study, a 30ms frame size is used (each frame has 480 samples) and each voice signal has a duration of 2 seconds with a 16kHz sampling rate, which means there are a total of 32,000 samples (2 × 16,000) per individual voice. <ul>
       <li> 2 seconds = 2,000ms, which means 32,000 samples have a total duration of 2,000ms. </li>
       <li> So that the number of samples each frame can contain:</li>
        <li>((32,000 samples × 30ms) ∕ 2000ms)) </li>
        <li>480 samples, each frame contains 480 samples.</li>
        </ul>
        <br/>
There is one major problem that may happen when dividing the voice signals into equal parts or frames, which is called spectral leakage. Endpoints in each frame are discontinuous, these discontinuities appear as high-frequency components but are not present in the original signal.
<br/>
To solve this problem, a method called windowing helps to eliminate the discontinuous samples at both ends of a frame. The popular windowing function used by different audio processing tools is called the “Hanning” window.
<br/>
After applying the Hanning windowing function, the endpoint of each frame is eliminated. However, these endpoints are lost and some of the contents of the audio are lost.
The solution is to take overlapping frames and overlap some (or half of a frame sample size is recommended) parts of the frame with the next frame to get the lost information at the endpoint of each frame due to applying the windowing function.
<br/>

Hop length is the number of samples considered before taking the next frame. In this study, a frame size of 480 samples with 240 overlapping samples (15ms overlapping in each frame) samples is used to divide the signals into an equal number of frames or segments. The total number of frames can be calculated using the following formula.
<ul>
 <li> Total number of frames=  (Total samples per second)/(hop-length)  +1, for 1second voice signal </li> 
 <li> Total number of frames=( (Total samples per second)/(hop-length))*2 +1, for 2 seconds voice signal.  Total number of frames=(16,000/240)*2 +1 = 134.  </li> 
</ul>

So, there are a total number of 134 frames in each voice signal.

## <b> N.B: if you need a dataset, contact me using LinkedIn </b> <a> linkedin.com/in/yeshiwas-dagnaw-alemu-961318172 </a>
