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
    model_id="ibm/granite-13b-instruct-v2",  # モデル名のスペースを修正
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    max_new_tokens=512,
    params=rag_gen_parameters,
)

from llama_index.readers.file import PyMuPDFReader


# Python の asyncio というライブラリを使って、自分だけの新しいイベントループを作り、両方が問題なく動くようにします。
import asyncio, nest_asyncio

nest_asyncio.apply()
loop = asyncio.get_event_loop()



# 以下のコードでは、すでに別のモデルを設定していた場合でも、WatsonxLLM クラスを使って granite-3-2-8b-instruct モデルで 上書き する形になります 。
watsonx_llm = WatsonxLLM(
	model_id="ibm/granite-3-2-8b-instruct",
	url="https://us-south.ml.cloud.ibm.com",
	project_id=os.getenv("WATSONX_PROJECT_ID"),
	max_new_tokens=512,
	params=rag_gen_parameters,
)

# 日本語に対応した Embedding モデル に変更
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
Settings.embed_model = HuggingFaceEmbedding(
	model_name="pkshatech/GLuCoSE-base-ja"
)

# トマトに関するpdfを読み込ませる。
from llama_index.readers.file import PyMuPDFReader
loader = PyMuPDFReader()
# pdf_doc_ja = loader.load(file_path="./docs/housetomato.pdf")
pdf_doc_ja = loader.load(file_path="./docs/housetomato.pdf")


from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=1024)

index = VectorStoreIndex.from_documents(
	pdf_doc_ja, transformations=[splitter],
	embed_model=Settings.embed_model
)





# ここから追加

# 新しいドキュメントと設定でインデックスを再構築
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512) # 日本語なのでチャンクサイズを調整
index_ja = VectorStoreIndex.from_documents(
    pdf_doc_ja, # 日本語のドキュメントを使用
    transformations=[splitter],
    embed_model=Settings.embed_model
)

# 新しいインデックスでリトリーバーを再構築
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


# クエリエンジンを新しいリトリーバーで再構築
from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine(retriever_ja)
print("日本語のドキュメントでインデックスとクエリエンジンを更新しました。")

# ベクトルを格納するためのインデックス を作成
## 今回は システムプロンプトを日本語 に変更し、トマトに関する指示を与えるようにする。
## 今回は「トマトの栽培方法」についての知識ベースとして使えるよう設定する。


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
    title="Granite",
    theme="soft",
    examples=[
        "トマトを家庭で育てるにはどうすればよいですか？",
        "トマトの栽培に最適な気候や土壌条件は何ですか？"
    ],
    type='messages'
)

# Gradioアプリケーションを起動します．
demo.launch(inline=True, share=False)
