# Usage
### 1. 必要フォルダの作成
```bash
mkdir docs/   # PDFファイルを格納するフォルダ
```
docs/ フォルダ内に，利用したい PDF（例: housetomato.pdf）を保存してください。
### 2. 必要なパッケージのインストール

以下を直接インストールするか，requirements.txt を利用します。

```
pip install fqdn getpass4 greenlet isoduration jsonpointer jupyterlab llama-index-embeddings-huggingface llama-index-llms-ibm llama-index-readers-file llama-index-retrievers-bm25 PyMuPDF tinycss2 uri -template webcolors sentencepiece
pip install ibm_watsonx_ai

```
また，requirements.txt に書かれたパッケージをまとめてインストール
```
pip install -r requirements.txt
```

### 3. 実行方法
```
python apps.py
```