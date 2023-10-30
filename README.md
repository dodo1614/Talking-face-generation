# Audio2Face
## Demo
1. Evaluate the trained model using:
```Shell
# preprocessing data
python process_data.py
# evaluate using GPU
python test.py --gpu 0 --video biden.mp4 --audio biden1.wav
