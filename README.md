# Talking face generation

1. Evaluate the trained model using:
```Shell
a. Download our pre-trained model (https://drive.google.com/file/d/1f7uS1zcSkg_pmQRA0yZzlGnqFlYFe70R/view?usp=sharing)
b. preprocessing data
python process_data.py
c. segment frames using https://github.com/zllrunning/face-parsing.PyTorch or unzip ./data/parsing.zip
d. evaluate using GPU
python test.py --gpu 0 --video biden.mp4 --audio biden1.wav

2. Results
https://github.com/dodo1614/Talking-face-generation/assets/33827796/5edb38d8-e597-4ab3-8364-e1c0609365cf

