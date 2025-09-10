# --- 1. 必要なライブラリのインポート ---
# (変更なし)
import os
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.readers.file import PyMuPDFReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ibm import WatsonxLLM
from llama_index.core.response_synthesizers import get_response_synthesizer
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.foundation_models import ModelInference
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    LLMMetadata,
    ChatMessage,
    MessageRole,
)

# --- 2. 環境変数の設定 ---
# (変更なし)
load_dotenv()
if not os.getenv("WATSONX_APIKEY") or not os.getenv("WATSONX_PROJECT_ID"):
    raise ValueError("環境変数 WATSONX_APIKEY と WATSONX_PROJECT_ID を.envファイルに設定してください。")

# --- 3. RAGの共通コンポーネントの構築 ---
# (変更なし)
print("RAG共通コンポーネントの構築を開始します...")
try:
    loader = PyMuPDFReader()
    pdf_doc_ja = loader.load(file_path="./docs/housetomato.pdf")
except Exception as e:
    raise FileNotFoundError(f"./docs/housetomato.pdf が見つかりません。: {e}")
embed_model = HuggingFaceEmbedding(model_name="pkshatech/GLuCoSE-base-ja")
Settings.embed_model = embed_model
splitter = SentenceSplitter(chunk_size=512)
index = VectorStoreIndex.from_documents(pdf_doc_ja, transformations=[splitter])
vector_retriever = index.as_retriever(similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=2)
print("クエリ生成用のLLM (Llama Maverick)を準備します...")
rag_gen_parameters_llama = {
    GenTextParamsMetaNames.DECODING_METHOD: "sample",
    GenTextParamsMetaNames.MIN_NEW_TOKENS: 150,
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 512,
    GenTextParamsMetaNames.TEMPERATURE: 0.5,
    GenTextParamsMetaNames.TOP_K: 5,
    GenTextParamsMetaNames.TOP_P: 0.7,
}
llm_llama = WatsonxLLM(
    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params=rag_gen_parameters_llama,
)
retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=4,
    num_queries=4,
    mode="reciprocal_rerank",
    llm=llm_llama,
    use_async=False,
    verbose=False,
)
print("RAG共通コンポーネントの構築が完了しました。")


# --- 4. 各LLMの準備とクエリエンジンの構築 ---
print("各モデルのクエリエンジンの構築を開始します...")
response_synthesizer_llama = get_response_synthesizer(llm=llm_llama, streaming=False)
query_engine_llama = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer_llama,
)
print("Llama用クエリエンジンが構築されました。")

# --- ★★★ 修正ポイント ★★★ ---
# WatsonxCustomLLMクラスに、不足していたメソッドを追加します
# This adds the missing required methods to the WatsonxCustomLLM class.
class WatsonxCustomLLM(CustomLLM):
    model_id: str = "openai/gpt-oss-120b"
    _model: ModelInference = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        return LLMMetadata(
            context_window=8192,
            num_output=1024,
            model_name=self.model_id,
        )

    def chat(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        watsonx_messages = [
            {"role": msg.role.value, "content": msg.content} 
            for msg in messages if msg.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]
        response = self._model.chat(messages=watsonx_messages)
        assistant_response = response["choices"][0]["message"]["content"]
        return CompletionResponse(
            text=assistant_response,
            message=ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response)
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        return self.chat([ChatMessage(role=MessageRole.USER, content=prompt)])

    # --- ↓↓ ここから2つのメソッドを追加 ↓↓ ---
    def stream_chat(self, messages: list[ChatMessage], **kwargs):
        # This method is required but not implemented for this use case.
        raise NotImplementedError("Streaming chat not supported")

    def stream_complete(self, prompt: str, **kwargs):
        # This method is required but not implemented for this use case.
        raise NotImplementedError("Streaming complete not supported")
    # --- ↑↑ ここまで追加 ↑↑ ---

llm_gpt = WatsonxCustomLLM() # これでエラーなく実行できるはずです
response_synthesizer_gpt = get_response_synthesizer(llm=llm_gpt, streaming=False)
query_engine_gpt = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer_gpt,
)
print("gpt-oss-120b用クエリエンジンが構築されました。")


# --- 5. Gradioインターフェースの定義 ---
# (変更なし)
def compare_models(question):
    print("Llamaモデルへの問い合わせを開始...")
    try:
        response_llama_obj = query_engine_llama.query(question)
        answer_llama = response_llama_obj.response
    except Exception as e:
        answer_llama = f"Llamaモデルでエラーが発生しました: {e}"
    print("Llamaモデルからの応答を取得しました。")
    print("gpt-oss-120bモデルへの問い合わせを開始...")
    try:
        response_gpt_obj = query_engine_gpt.query(question)
        answer_gpt = response_gpt_obj.response
    except Exception as e:
        answer_gpt = f"gpt-oss-120bモデルでエラーが発生しました: {e}"
    print("gpt-oss-120bモデルからの応答を取得しました。")
    return answer_llama, answer_gpt
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# Llama (Maverick) vs gpt-oss-120b モデル回答比較")
    gr.Markdown("同じ質問を2つの異なるLLMに投げかけ、回答を比較します。")
    with gr.Row():
        question_box = gr.Textbox(
            label="質問を入力してください",
            placeholder="例: トマトを家庭で育てるにはどうすればよいですか？",
            scale=4
        )
        submit_btn = gr.Button("送信", variant="primary", scale=1)
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Llama (Maverick)")
            answer_llama_box = gr.Textbox(label="Llamaの回答", lines=20, interactive=False)
        with gr.Column():
            gr.Markdown("## gpt-oss-120b")
            answer_gpt_box = gr.Textbox(label="gpt-oss-120bの回答", lines=20, interactive=False)
    gr.Examples(
        examples=[
            "トマトを家庭で育てるにはどうすればよいですか？",
            "トマトの栽培に最適な気候や土壌条件は何ですか？",
            "トマトの主な病気と、その対策について教えてください。"
        ],
        inputs=question_box
    )
    submit_btn.click(
        fn=compare_models,
        inputs=question_box,
        outputs=[answer_llama_box, answer_gpt_box]
    )

# --- 6. アプリケーションの起動 ---
# (変更なし)
if __name__ == "__main__":
    print("チャットボットを起動します。以下のURLにアクセスしてください。")
    demo.launch()
