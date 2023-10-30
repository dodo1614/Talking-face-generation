# Audio2Face
## Demo
1. Evaluate the trained model using:
```Shell
a. Download [our pre-trained model](https://drive.google.com/file/d/1f7uS1zcSkg_pmQRA0yZzlGnqFlYFe70R/view?usp=sharing)
b. preprocessing data
python process_data.py
c. evaluate using GPU
python test.py --gpu 0 --video biden.mp4 --audio biden1.wav
