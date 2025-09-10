import os
import gradio as gr
from dotenv import load_dotenv

# --- LlamaIndexとwatsonxのライブラリをインポート ---
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.readers.file import PyMuPDFReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

# --- LlamaIndexのカスタムLLMを作成するために必要なクラスをインポート ---
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    LLMMetadata,
    ChatMessage,
    MessageRole,
)

# --------------------------------------------------------------------------
# 1. 動作確認済みのModelInferenceをラップするカスタムLLMクラスの定義
# --------------------------------------------------------------------------

class WatsonxCustomLLM(CustomLLM):
    """
    gpt4o.pyで動作確認が取れたModelInferenceクラスを
    LlamaIndexで使えるようにするためのカスタムラッパークラス。
    """
    model_id: str = "openai/gpt-oss-120b"
    _model: ModelInference = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # gpt4o.pyと同様にModelInferenceオブジェクトを初期化します
        parameters = {
            GenTextParamsMetaNames.DECODING_METHOD: "sample",
            GenTextParamsMetaNames.MIN_NEW_TOKENS: 150,
            GenTextParamsMetaNames.MAX_NEW_TOKENS: 1024,
            GenTextParamsMetaNames.TEMPERATURE: 0.7,
            GenTextParamsMetaNames.TOP_K: 50,
            GenTextParamsMetaNames.TOP_P: 0.9,
        }
        
        self._model = ModelInference(
            model_id=self.model_id,
            params=parameters,
            credentials={
                "apikey": os.getenv("WATSONX_APIKEY"),
                "url": "https://us-south.ml.cloud.ibm.com"
            },
            project_id=os.getenv("WATSONX_PROJECT_ID")
        )

    @property
    def metadata(self) -> LLMMetadata:
        # LlamaIndexが必要とするメタデータを手動で設定します
        return LLMMetadata(
            context_window=8192,  # モデルのコンテキストウィンドウサイズ
            num_output=1024,      # 最大出力トークン数
            model_name=self.model_id,
        )

    def chat(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        # LlamaIndexのメッセージ形式をwatsonxの形式に変換します
        watsonx_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]
            
        # モデルを呼び出します
        response = self._model.chat(messages=watsonx_messages)
        
        # 応答をLlamaIndexの形式に変換して返します
        assistant_response = response["choices"][0]["message"]["content"]
        return CompletionResponse(
            text=assistant_response,  # 必須フィールド'text'を追加
            message=ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response)
        )
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        # chatメソッドを呼び出す形で実装
        return self.chat([ChatMessage(role=MessageRole.USER, content=prompt)])

    # ストリーミングは未実装
    def stream_chat(self, messages: list[ChatMessage], **kwargs):
        raise NotImplementedError("Streaming not implemented")

    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Streaming not implemented")

# --------------------------------------------------------------------------
# 2. 環境設定とモデルの初期化
# --------------------------------------------------------------------------

# .envファイルから環境変数を読み込みます
load_dotenv()

# APIキーとプロジェクトIDの存在を確認します
if not os.getenv("WATSONX_APIKEY") or not os.getenv("WATSONX_PROJECT_ID"):
    raise ValueError("環境変数 WATSONX_APIKEY と WATSONX_PROJECT_ID を.envファイルに設定してください。")

# --- LlamaIndex全体で使用するモデルを設定 ---
# 作成したカスタムLLMクラスを初期化します
llm = WatsonxCustomLLM()
# 日本語のEmbeddingモデルを初期化します
embed_model = HuggingFaceEmbedding(model_name="pkshatech/GLuCoSE-base-ja")

# LlamaIndexのグローバル設定に各モデルをセットします
Settings.llm = llm
Settings.embed_model = embed_model

# --------------------------------------------------------------------------
# 3. PDFの読み込みからクエリエンジンの構築 (ここは変更なし)
# --------------------------------------------------------------------------

loader = PyMuPDFReader()
pdf_doc_ja = loader.load(file_path="./docs/housetomato.pdf") 

splitter = SentenceSplitter(chunk_size=512)
index = VectorStoreIndex.from_documents(pdf_doc_ja, transformations=[splitter])

vector_retriever = index.as_retriever(similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=2)

query_gen_prompt_str = (
    "あなたは、1つの入力クエリに基づいて複数の検索クエリを生成する有能なアシストです。\n"
    "{num_queries}個の検索クエリを、1行につき1つずつ生成してください。\n"
    "以下のクエリに関連する検索クエリを生成してください：\n\n"
    "クエリ: {query}\n"
    "検索クエリ:\n"
)

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=4,
    num_queries=4,
    mode="reciprocal_rerank",
)

query_engine = RetrieverQueryEngine(retriever)

# --------------------------------------------------------------------------
# 4. Gradio UIの構築と起動 (ここは変更なし)
# --------------------------------------------------------------------------

def chat_function(message, history):
    try:
        response_obj = query_engine.query(message)
        result = response_obj.response
        
        source_nodes = response_obj.source_nodes
        if source_nodes:
            result += "\n\n--- ソース ---\n"
            unique_sources = set()
            for node_with_score in source_nodes:
                node = node_with_score.node
                page_num = node.metadata.get('page_label', 'N/A')
                file_name = node.metadata.get('file_name', 'N/A')
                
                source_id = f"{file_name}-p{page_num}"
                if source_id not in unique_sources:
                    result += f"- ファイル: {file_name}, ページ: {page_num}\n"
                    content_preview = node.get_content()[:100].replace('\n', ' ')
                    result += f"  内容: {content_preview}...\n"
                    unique_sources.add(source_id)
        return result
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

demo = gr.ChatInterface(
    fn=chat_function,
    title="gpt-oss-120b",
    description="PDFドキュメントに関する質問をしてください。",
    theme="soft",
    examples=[
        "トマトを家庭で育てるにはどうすればよいですか？",
        "トマトの栽培に最適な気候や土壌条件は何ですか？"
    ]
)

if __name__ == "__main__":
    print("チャットボットを起動します。URLにアクセスしてください。")
    demo.launch()
