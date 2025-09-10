import os
import gradio as gr
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.readers.file import PyMuPDFReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.ibm import WatsonxLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

# --------------------------------------------------------------------------
# 1. watsonx と Embedding モデルのセットアップ
# --------------------------------------------------------------------------

# watsonxのAPIキーとプロジェクトIDを設定します
watsonx_api_key = "0i-_-6pigerNnnRaU8_oiybRZz_UxMQuBHpE_copxSdw"
os.environ["WATSONX_APIKEY"] = watsonx_api_key

watsonx_project_id = "b596c884-f867-4771-afcc-f9fd10dae1a4"
os.environ["WATSONX_PROJECT_ID"] = watsonx_project_id

# LLMの生成パラメータを定義します
rag_gen_parameters = {
    GenTextParamsMetaNames.DECODING_METHOD: "sample",
    GenTextParamsMetaNames.MIN_NEW_TOKENS: 150,
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 512,
    GenTextParamsMetaNames.TEMPERATURE: 0.5,
    GenTextParamsMetaNames.TOP_K: 5,
    GenTextParamsMetaNames.TOP_P: 0.7,
}

# watsonxのLLMを初期化します
watsonx_llm = WatsonxLLM(
    model_id="openai/gpt-oss-120b",  # モデルIDをopenai/gpt-oss-120bに変更
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params=rag_gen_parameters,
)

# 日本語に対応したEmbeddingモデルを設定します
embed_model = HuggingFaceEmbedding(model_name="pkshatech/GLuCoSE-base-ja")

# LlamaIndex全体で使用するLLMとEmbeddingモデルを設定します
Settings.llm = watsonx_llm
Settings.embed_model = embed_model


# --------------------------------------------------------------------------
# 2. PDFドキュメントの読み込みとインデックスの構築
# --------------------------------------------------------------------------

# PDFファイルを読み込みます
loader = PyMuPDFReader()
# PDFファイルへのパスを指定してください
pdf_doc_ja = loader.load(file_path="./docs/housetomato.pdf") 

# テキストを適切なサイズのチャンクに分割するスプリッターを定義します
splitter = SentenceSplitter(chunk_size=512)

# ドキュメントからベクトルインデックスを構築します
index = VectorStoreIndex.from_documents(
    pdf_doc_ja,
    transformations=[splitter],
)


# --------------------------------------------------------------------------
# 3. リトリーバーとクエリエンジンの構築
# --------------------------------------------------------------------------

# 2種類のリトリーバーを準備します (Vector + BM25)
vector_retriever = index.as_retriever(similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=2)

# クエリを複数生成するためのプロンプトを定義します
query_gen_prompt_str = (
    "あなたは、1つの入力クエリに基づいて複数の検索クエリを生成する有能なアシスタントです。\n"
    "{num_queries}個の検索クエリを、1行につき1つずつ生成してください。\n"
    "以下のクエリに関連する検索クエリを生成してください：\n\n"
    "クエリ: {query}\n"
    "検索クエリ:\n"
)

# 複数のリトリーバーを統合するQueryFusionRetrieverを構築します
retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=4,
    num_queries=4,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=False,
    query_gen_prompt=query_gen_prompt_str,
)

# RAGの応答を生成するクエリエンジンを構築します
query_engine = RetrieverQueryEngine(retriever)


# --------------------------------------------------------------------------
# 4. GradioによるチャットUIの構築と起動
# --------------------------------------------------------------------------

# Gradioのチャットボットが呼び出す関数を定義します
def chat_function(message, history):
    try:
        # query_engine.queryメソッドで応答を生成します
        response_obj = query_engine.query(message)
        result = response_obj.response
        
        # 参照元の情報を応答に追加します
        source_nodes = response_obj.source_nodes
        if source_nodes:
            result += "\n\n--- ソース ---\n"
            # 重複するソースを表示しないように管理します
            unique_sources = set()
            for node_with_score in source_nodes:
                node = node_with_score.node
                # メタデータからページ番号とファイル名を取得します
                page_num = node.metadata.get('page_label', 'N/A')
                file_name = node.metadata.get('file_name', 'N/A')
                
                source_id = f"{file_name}-p{page_num}"
                if source_id not in unique_sources:
                    result += f"- ファイル: {file_name}, ページ: {page_num}\n"
                    # 参照された内容の冒頭部分を表示します
                    content_preview = node.get_content()[:100].replace('\n', ' ')
                    result += f"  内容: {content_preview}...\n"
                    unique_sources.add(source_id)
                    
        return result
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

# Gradioのチャットインターフェースを作成します
demo = gr.ChatInterface(
    fn=chat_function,
    title="トマトマスター🍅",
    description="PDFドキュメントに関する質問をしてください。",
    theme="soft",
    examples=[
        "トマトを家庭で育てるにはどうすればよいですか？",
        "トマトの栽培に最適な気候や土壌条件は何ですか？"
    ]
)

# Gradioアプリを起動します
if __name__ == "__main__":
    print("チャットボットを起動します。URLにアクセスしてください。")
    demo.launch()
