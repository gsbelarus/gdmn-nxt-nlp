cuda 11.3

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install -U pip setuptools wheel
pip install -U 'spacy[cuda113,transformers,lookups]'

python -m spacy download en_core_web_trf
python -m spacy download pl_core_news_lg
python -m spacy download ru_core_news_lg

pip install jupyterlab
pip install notebook
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension