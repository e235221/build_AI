# 必要なライブラリをインポート
import os
from getpass import getpass

# watsonx.aiのAPIキーとプロジェクトIDを環境変数に設定
watsonx_api_key = "0i-_-6pigerNnnRaU8_oiybRZz_UxMQuBHpE_copxSdw"
os.environ["WATSONX_APIKEY"] = watsonx_api_key

watsonx_project_id = "b596c884-f867-4771-afcc-f9fd10dae1a4"
os.environ["WATSONX_PROJECT_ID"] = watsonx_project_id

# LlamaIndexとIBM WatsonX AIから必要なクラスをインポート
from llama_index.llms.ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

# LLMの生成パラメータを設定
rag_gen_parameters = {
    GenTextParamsMetaNames.DECODING_METHOD: "sample",  # サンプリングを使用してテキストを生成
    GenTextParamsMetaNames.MIN_NEW_TOKENS: 150,       # 生成する最小トークン数
    GenTextParamsMetaNames.TEMPERATURE: 0.5,         # 生成テキストのランダム性を制御
    GenTextParamsMetaNames.TOP_K: 5,                 # 上位K個のトークンからサンプリング
    GenTextParamsMetaNames.TOP_P: 0.7                  # 上位Pの確率質量からサンプリング
}

# WatsonxLLMを初期化
watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-13b-instruct-v2",  # 使用するモデルのID
    url="https://us-south.ml.cloud.ibm.com", # watsonx.aiのエンドポイントURL
    project_id=os.getenv("WATSONX_PROJECT_ID"), # 環境変数からプロジェクトIDを取得
    max_new_tokens=512, # 生成する最大トークン数
    params=rag_gen_parameters, # 上で定義した生成パラメータを適用
)

# PDFファイルを読み込むためのPyMuPDFReaderをインポート
from llama_index.readers.file import PyMuPDFReader

# Jupyter Notebookのような環境でasyncioのイベントループがネストされる問題を解決するためにnest_asyncioを適用します。
# これにより、既存のイベントループ内で新しいイベントループを実行できるようになります。
import asyncio, nest_asyncio

# nest_asyncioを適用
nest_asyncio.apply()
# 現在のイベントループを取得
loop = asyncio.get_event_loop()

# checkpoint
# トークナイザがSentencePieceを使って正常にロードできるか確認
from transformers import AutoTokenizer
# 日本語の事前学習済みモデルのトークナイザをロード
tok = AutoTokenizer.from_pretrained("pkshatech/GLuCoSE-base-ja")
# トークナイザの型と"OK"メッセージを出力して、ロードが成功したことを確認
print(type(tok), "OK")

# HuggingFaceの埋め込みモデルとLlamaIndexのグローバル設定をインポート
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# グローバル設定で、使用する埋め込みモデルをHuggingFaceのモデルに設定
# これにより、以降の処理でこの埋め込みモデルがデフォルトで使用される
Settings.embed_model = HuggingFaceEmbedding(
    model_name="pkshatech/GLuCoSE-base-ja"  # 日本語のテキストに適した埋め込みモデル
)


watsonx_llm = WatsonxLLM(
	model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
	url="https://us-south.ml.cloud.ibm.com",
	project_id=os.getenv("WATSONX_PROJECT_ID"),
	max_new_tokens=512,
	params=rag_gen_parameters,
)

# トマトに関するpdfを読み込ませる。
from llama_index.readers.file import PyMuPDFReader
loader = PyMuPDFReader()
# pdf_doc_ja = loader.load(file_path="./docs/housetomato.pdf")
pdf_doc_ja = loader.load(file_path="./docs/housetomato.pdf")


from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
# テキストを1024文字のチャンクに分割するスプリッターを定義
splitter = SentenceSplitter(chunk_size=1024)

# PDFドキュメントからベクトルストアインデックスを作成
index = VectorStoreIndex.from_documents(
	pdf_doc_ja, transformations=[splitter],
	embed_model=Settings.embed_model # グローバル設定の埋め込みモデルを使用
)

# 日本語に対応した Embedding モデル に変更
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
Settings.embed_model = HuggingFaceEmbedding(
	model_name="pkshatech/GLuCoSE-base-ja"
)

# ここから追加

# 新しいドキュメントと設定でインデックスを再構築
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

# 日本語のテキストに適したチャンクサイズでスプリッターを再定義
splitter = SentenceSplitter(chunk_size=512) # 日本語なのでチャンクサイズを調整
# 日本語ドキュメントと新しい設定でベクトルストアインデックスを再構築
index_ja = VectorStoreIndex.from_documents(
    pdf_doc_ja, # 日本語のドキュメントを使用
    transformations=[splitter],
    embed_model=Settings.embed_model
)

# 新しいインデックスでリトリーバーを再構築
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# ベクトル検索用のリトリーバーを作成（類似度上位2件を取得）
vector_retriever_ja = index_ja.as_retriever(similarity_top_k=2)
# BM25（キーワードベース）検索用のリトリーバーを作成（類似度上位2件を取得）
bm25_retriever_ja = BM25Retriever.from_defaults(
    docstore=index_ja.docstore,
    similarity_top_k=2
)

# Query Fusion Retrieverが使用するクエリ生成のプロンプトを定義
# このプロンプトは、元のクエリから複数の異なる検索クエリを生成するようにLLMに指示する
query_gen_prompt_str = (
    "あなたは、1つの入力クエリに基づいて複数の検索クエリを生成する有能なアシスタントです。\n"
    "{num_queries}個の検索クエリを、1行につき1つずつ生成してください。\n"
    "以下のクエリに関連する検索クエリを生成してください：\n"
    "\n"
    "クエリ: {query}\n"
    "検索クエリ:\n"
)


# QueryFusionRetrieverとBM25Retrieverをインポート
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

# グローバル設定でLLMをwatsonx_llmに設定
Settings.llm = watsonx_llm

# ベクトル検索用のリトリーバーを作成
vector_retriever = index.as_retriever(similarity_top_k=2)

# BM25検索用のリトリーバーを作成
bm25_retriever = BM25Retriever.from_defaults(
	docstore=index.docstore,
	similarity_top_k=2
)

# 複数のリトリーバーを組み合わせるQueryFusionRetrieverを作成
retriever = QueryFusionRetriever(
	[vector_retriever, bm25_retriever], # ベクトル検索とBM25検索を組み合わせる
	similarity_top_k=4, # 最終的に返すドキュメントの数
	num_queries=4,  # 生成する検索クエリの数
	mode="reciprocal_rerank", # 検索結果をランク付けするモード
	use_async=False, # 同期的に実行
	verbose=False, # 詳細なログ出力を無効化
	query_gen_prompt=query_gen_prompt_str  # 上で定義したクエリ生成プロンプトを使用
)


# 日本語設定でQueryFusionRetrieverを初期化
retriever_ja = QueryFusionRetriever(
    [vector_retriever_ja, bm25_retriever_ja], # 日本語用のリトリーバーを使用
    similarity_top_k=4, # 最終的に返すドキュメントの数
    num_queries=4, # 生成する検索クエリの数
    mode="reciprocal_rerank", # ランク付けモード
    use_async=False, # 同期的に実行
    verbose=False, # 詳細なログ出力を無効化
    query_gen_prompt=query_gen_prompt_str # 日本語プロンプトを使用
)


# クエリエンジンを新しいリトリーバーで再構築
from llama_index.core.query_engine import RetrieverQueryEngine
# 日本語用のリトリーバーを使ってクエリエンジンを作成
query_engine = RetrieverQueryEngine(retriever_ja)
print("日本語のドキュメントでインデックスとクエリエンジンを更新しました。")

# ベクトルを格納するためのインデックス を作成
## 今回は システムプロンプトを日本語 に変更し、トマトに関する指示を与えるようにする。
## 今回は「トマトの栽培方法」についての知識ベースとして使えるよう設定する。


# RetrieverQueryEngineをインポート
from llama_index.core.query_engine import RetrieverQueryEngine
# 上で作成したretriever（英語設定）を使用してクエリエンジンを作成
query_engine = RetrieverQueryEngine(retriever)

# Gradioライブラリをインポート
import gradio as gr

def chat_function(message, history):
    """
    チャットメッセージを処理し、応答を生成します。
    """
    try:
        # クエリエンジンでメッセージ（質問）を処理します。
        response_obj = query_engine.query(message)
        # 応答オブジェクトからテキスト部分を取得します。
        response_text = response_obj.response
    except Exception as e:
        # エラーが発生した場合、エラーメッセージを返します。
        response_text = f"エラーが発生しました: {e}"
    return response_text

# GradioのChatInterfaceを作成
demo = gr.ChatInterface(
    fn=chat_function, # チャットの応答を生成する関数
    title="Llama", # インターフェースのタイトル
    theme="soft", # UIのテーマ
    examples=[ # ユーザーに示す質問の例
        "トマトを家庭で育てるにはどうすればよいですか？",
        "トマトの栽培に最適な気候や土壌条件は何ですか？"
    ],
    type='messages' # メッセージ形式のインターフェース
)

# Gradioアプリケーションを起動します。
demo.launch(inline=True, share=False) # インラインで表示し、共有リンクは作成しない
