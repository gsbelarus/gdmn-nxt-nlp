python 3.10

visual studio 2019 community edition + c++ workload:
https://www.techspot.com/downloads/7241-visual-studio-2019.html

cuda 11.3:
https://developer.nvidia.com/cuda-11.3.0-download-archive

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install -U pip setuptools wheel
pip install -U 'spacy[cuda113,transformers,lookups]'

python -m spacy download en_core_web_trf
python -m spacy download pl_core_news_lg
python -m spacy download ru_core_news_lg

pip install jupyterlab
pip install notebook
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension