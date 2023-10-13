# Computer-Commands-and-Control-Model-Using-Amharic_voice_commands--in-CNN-and-LSTM-Algorithms
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

## <b>Data Collection</b>
To develop a model, the required type of data is audio or voice. The voice data is collected using smartphones via voice recorder applications. The built model is speaker-independent, which means it can recognize speech commands regardless of the speaker. So, to solve acoustic-related problems (such as accents and pronunciations), the voices are collected from 35 different Amharic language speakers (20 males and 15 females in the age range of 21-28) at least five times per command. The speech commands are recorded at a sample rate of 16kHhz with the mono channel and then they are stored in .wav audio file format. Most of the voices are recorded in less-noisy environments (i.e. at home and the office, because the acoustic environment and transduction equipment have a great effect). A total of 7000 voices have been collected (all voices are recorded using mobile phones).

To feed data to the CNN and LSTm models, two audio features (MFCCs and Mel-Spectrogram) are extracted from the raw audio files.
These two features 
