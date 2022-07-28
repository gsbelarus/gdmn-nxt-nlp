## Setup with CUDA enabled GPU

1. python 3.10
2. visual studio 2019 community edition + c++ workload: https://www.techspot.com/downloads/7241-visual-studio-2019.html
3. cuda 11.3: https://developer.nvidia.com/cuda-11.3.0-download-archive
4. `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
5. `pip install -U pip setuptools wheel`
6. `pip install -U "spacy[cuda113,transformers,lookups]"`
7. `python -m spacy download en_core_web_trf`
8. `python -m spacy download pl_core_news_lg`
9. `python -m spacy download ru_core_news_lg`
10. `pip install jupyterlab`
11. `pip install notebook`
12. `pip install ipywidgets`
13. `jupyter nbextension enable --py widgetsnbextension`
