pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install -U "spacy[cuda116,transformers,lookups]"
python -m spacy download en_core_web_trf
python -m spacy download pl_core_news_lg
python -m spacy download ru_core_news_lg
pip install -U jupyterlab
pip install -U notebook
pip install -U ipywidgets
jupyter nbextension enable --py widgetsnbextension
pip install -U spacy_langdetect
pip install -U python-dotenv
pip install -U spacy_universal_sentence_encoder
pip install -U sentencepiece
pip install -U matplotlib