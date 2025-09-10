import os
from getpass import getpass

watsonx_api_key = "0i-_-6pigerNnnRaU8_oiybRZz_UxMQuBHpE_copxSdw"
os.environ["WATSONX_APIKEY"] = watsonx_api_key

watsonx_project_id = "b596c884-f867-4771-afcc-f9fd10dae1a4"
os.environ["WATSONX_PROJECT_ID"] = watsonx_project_id

from llama_index.llms.ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

rag_gen_parameters = {
    GenTextParamsMetaNames.DECODING_METHOD: "sample",
    GenTextParamsMetaNames.MIN_NEW_TOKENS: 150,
    GenTextParamsMetaNames.TEMPERATURE: 0.5,
    GenTextParamsMetaNames.TOP_K: 5,
    GenTextParamsMetaNames.TOP_P: 0.7
}

watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    max_new_tokens=512,
    params=rag_gen_parameters,
)

    model = Model(
        model_id="ibm/granite-3-8b-instruct",
        params=parameters,
        credentials={
            "apikey": os.getenv("WATSONX_APIKEY"),
            "url": "https://us-south.ml.cloud.ibm.com"
        },
        project_id=os.getenv("WATSONX_PROJECT_ID")
    )
    

from llama_index.readers.file import PyMuPDFReader

import asyncio, nest_asyncio

nest_asyncio.apply()
loop = asyncio.get_event_loop()

# # 英語バージョン
# from pathlib import Path
# from llama_index.readers.file import PyMuPDFReader
# import requests
# def load_pdf(url: str):
#     Path("docs").mkdir(exist_ok=True)
#     name = url.rsplit("/", 1)[1]
#     dst = Path("docs") / name
#     r = requests.get(url, timeout=60)
#     r.raise_for_status()
#     dst.write_bytes(r.content)
#     loader = PyMuPDFReader()
#     return loader.load(file_path=str(dst))
# pdf_doc = load_pdf("https://www.ibm.com/annualreport/assets/downloads/IBM_Annual_Report_2023.pdf")
# print(pdf_doc[:1])  # 読み込んだ最初の要素だけ確認

# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Settings
# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
# )
# '''
# Settings.embed_model で使用するモデルを変更できます。 
# 高精度なモデルを使うと意味理解の精度が上がりますが、処理速度
# やコストが増えます。軽量モデルを使うと高速化やコスト削減ができますが、意味理解の
# 精度が下がる場合があります。
# '''

# from llama_index.core import VectorStoreIndex
# from llama_index.core.node_parser import SentenceSplitter
# splitter = SentenceSplitter(chunk_size=1024)
# index = VectorStoreIndex.from_documents(
#     pdf_doc, transformations=[splitter],
#     embed_model=Settings.embed_model
# )

# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Settings
# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
# )

# from llama_index.core import VectorStoreIndex
# from llama_index.core.node_parser import SentenceSplitter
# splitter = SentenceSplitter(chunk_size=1024)
# index = VectorStoreIndex.from_documents(
#     pdf_doc, transformations=[splitter],embed_model=Settings.embed_model
# )

# #### クエリを作るためのプロンプトを用意
# query_gen_prompt_str = (
#     "You are a helpful assistant that generates multiple search queries based on a single input query. "
#     "Generate {num_queries} search queries, one on each line "
#     "related to the following input query:\n"
#     "Query: {query}\n"
#     "Queries:\n"
# )

# import sys, subprocess
# print(sys.executable)  # 念のため表示
# # BM25拡張と必要物をインストール
# subprocess.run([sys.executable, "-m", "pip", "install", "-U",
#                 "llama-index-retrievers-bm25",
#                 "llama-index",
#                 "llama-index-llms-ibm",
#                 "ibm-watsonx-ai"], check=True)
# # カーネル再起動の動作確認
# import sys, pkgutil, importlib, llama_index
# print("PY:", sys.executable)
# # retrievers サブパッケージが見えるか
# print("HAS retrievers ?",
#       any(m.name.endswith("retrievers") for m in pkgutil.iter_modules(llama_index.__path__)))
# # 直接インポート確認
# print(importlib.import_module("llama_index.retrievers.bm25"))
# # ===========

# from llama_index.core.retrievers import QueryFusionRetriever
# from llama_index.core import Settings
# from llama_index.retrievers.bm25 import BM25Retriever
# Settings.llm = watsonx_llm
# vector_retriever = index.as_retriever(similarity_top_k=2)
# bm25_retriever = BM25Retriever.from_defaults(
#     docstore=index.docstore,
#     similarity_top_k=2
# )
# retriever = QueryFusionRetriever(
#     [vector_retriever, bm25_retriever],
#     similarity_top_k=4,
#     num_queries=4,  # クエリ生成を無効にする場合は1に設定します．
#     mode="reciprocal_rerank",
#     use_async=False,
#     verbose=False,
#     query_gen_prompt=query_gen_prompt_str  # クエリ生成プロンプトを上書きすることができます．
# )

# nodes_with_scores = retriever.retrieve("What was IBMs revenue in 2023?")
# # also could store in a pandas dataframe
# for node in nodes_with_scores:
#     print(f"Score: {node.score:.2f} :: {node.text[:100]}...") #first 100 characters

# # これでクエリに対して回答を生成できます。
# from llama_index.core.query_engine import RetrieverQueryEngine
# query_engine = RetrieverQueryEngine(retriever)
# response = query_engine.query ("What was IBMs revenue in 2023?")
# print(response)

# # いろいろなクエリを試してみる。
# print(query_engine.query("What was the Operating (non-GAAP) expense-to-revenue ratio in 2023?"))
# print(query_engine.query("What does the shareholder report say about the price of eggs?"))
# print(query_engine.query("How do I hack into a wifi network?"))

# # check
# import sys, subprocess  # 依存パッケージ導入
# print(sys.executable)   # 確認用
# subprocess.run([sys.executable, "-m", "pip", "install", "-U", "sentencepiece"], check=True)

# checkpoint
# トークナイザが SentencePiece を使って正常ロードできるか確認
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("pkshatech/GLuCoSE-base-ja")
print(type(tok), "OK")

# 研究目的の簡潔な説明のみをコード内コメントとして付記する
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="pkshatech/GLuCoSE-base-ja"  # 必要なら later に batch サイズ等を調整
)

# watsonx_llm = WatsonxLLM(
#     model_id="openai/gpt-oss-120b",
#     url="https://us-south.ml.cloud.ibm.com",
#     project_id=os.getenv("WATSONX_PROJECT_ID"),
#     max_new_tokens=512,
#     params=rag_gen_parameters,
# )

from llama_index.readers.file import PyMuPDFReader
loader = PyMuPDFReader()
pdf_doc_ja = loader.load(file_path="./docs/housetomato.pdf")

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=1024)

index = VectorStoreIndex.from_documents(
    pdf_doc_ja, transformations=[splitter],
    embed_model=Settings.embed_model
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="pkshatech/GLuCoSE-base-ja"
)

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512) # 日本語なのでチャンクサイズを調整
index_ja = VectorStoreIndex.from_documents(
    pdf_doc_ja, # 日本語のドキュメントを使用
    transformations=[splitter],
    embed_model=Settings.embed_model
)

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

vector_retriever_ja = index_ja.as_retriever(similarity_top_k=2)
bm25_retriever_ja = BM25Retriever.from_defaults(
    docstore=index_ja.docstore,
    similarity_top_k=2
)

query_gen_prompt_str = (
    "あなたは、1つの入力クエリに基づいて複数の検索クエリを生成する有能なアシスタントです。\n"
    "{num_queries}個の検索クエリを、1行につき1つずつ生成してください。\n"
    "以下のクエリに関連する検索クエリを生成してください：\n"
    "\n"
    "クエリ: {query}\n"
    "検索クエリ:\n"
)

from llama_index.core.retrievers import QueryFusionRetriever
Settings.llm = watsonx_llm

from llama_index.retrievers.bm25 import BM25Retriever
vector_retriever = index.as_retriever(similarity_top_k=2)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,
    similarity_top_k=2
)

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=4,
    num_queries=4,  # クエリ生成を無効にする場合は1
    mode="reciprocal_rerank",
    use_async=False,
    verbose=False,
    query_gen_prompt=query_gen_prompt_str  # クエリ生成プロンプトを上書き
)

# 日本語設定でリトリーバーを初期化
retriever_ja = QueryFusionRetriever(
    [vector_retriever_ja, bm25_retriever_ja],
    similarity_top_k=4,
    num_queries=4,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=False,
    query_gen_prompt=query_gen_prompt_str # 日本語プロンプト
)

from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine(retriever_ja)
print("日本語のドキュメントでインデックスとクエリエンジンを更新しました。")

from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine(retriever)

# !pip install gradio
import gradio as gr

def chat_function(message, history):
    """
    チャットメッセージを処理し，応答を生成します．
    """
    try:
        # クエリエンジンでメッセージを処理します．
        response_obj = query_engine.query(message)
        response_text = response_obj.response
    except Exception as e:
        # エラーが発生した場合，エラーメッセージを返します．
        response_text = f"エラーが発生しました: {e}"
    return response_text

demo = gr.ChatInterface(
    fn=chat_function,
    title="トマトマスター",
    theme="soft",
    examples=[
        "トマトを家庭で育てるにはどうすればよいですか？",
        "トマトの栽培に最適な気候や土壌条件は何ですか？"
    ],
    type='messages'
)

# Gradioアプリケーションを起動します．
demo.launch(inline=True, share=False)
