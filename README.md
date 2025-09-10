# 概要

このプロジェクトは IBM watsonxを用いたLlamaとGPT modelを利用したチャットアプリケーションです。Gradio を用いてブラウザ上から対話できます。

# Useage

```
pip install fqdn getpass4 greenlet isoduration jsonpointer jupyterlab llama-index-embeddings-huggingface llama-index-llms-ibm llama-index-readers-file llama-index-retrievers-bm25 PyMuPDF tinycss2 uri -template webcolors sentencepiece
pip install ibm_watsonx_ai

```

requirements.txt に書かれたパッケージをまとめてインストール
```
pip install -r requirements.txt
```

実行する。
```
python chat.py
```
