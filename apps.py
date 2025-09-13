import os
import asyncio
import nest_asyncio
from typing import Tuple
from dotenv import load_dotenv

#  環境設定 -
# 注: APIキーやプロジェクトIDは .env から読み込む設計に変更する．
load_dotenv(dotenv_path=".env", override=False)  # 既存の環境変数が優先される．
WX_APIKEY = os.getenv("WATSONX_APIKEY") or os.getenv("WATSONX_API_KEY")
WX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

if not WX_APIKEY or not WX_PROJECT_ID:
    raise RuntimeError(
        "`.env` に WATSONX_APIKEY と WATSONX_PROJECT_ID を記載してください。または環境変数に設定してください．"
    )

#  ライブラリ類 
# 研究目的の簡潔なコメント: LlamaIndex と watsonx.ai によるRAGクエリ比較のUIを提供する.
from llama_index.llms.ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

import gradio as gr

#  Notebook/REPL互換 
nest_asyncio.apply()
_ = asyncio.get_event_loop()

#  生成パラメータ 
RAG_GEN_PARAMS = {
    GenTextParamsMetaNames.DECODING_METHOD: "sample",
    GenTextParamsMetaNames.MIN_NEW_TOKENS: 150,
    GenTextParamsMetaNames.TEMPERATURE: 0.5,
    GenTextParamsMetaNames.TOP_K: 5,
    GenTextParamsMetaNames.TOP_P: 0.7,
}

#  LLMの初期化（2モデル） 
# 研究目的の簡潔なコメント: Granite と Llama の2系統LLMを個別初期化する.
GRANITE_MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-2-8b-instruct")
LLAMA_MODEL_ID = os.getenv(
    "LLAMA_MODEL_ID", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
)

granite_llm = WatsonxLLM(
    model_id=GRANITE_MODEL_ID,
    url=WX_URL,
    project_id=WX_PROJECT_ID,
    max_new_tokens=512,
    params=RAG_GEN_PARAMS,
)

llama_llm = WatsonxLLM(
    model_id=LLAMA_MODEL_ID,
    url=WX_URL,
    project_id=WX_PROJECT_ID,
    max_new_tokens=512,
    params=RAG_GEN_PARAMS,
)

# QueryFusionRetrieverのサブクエリ生成で使用するLLMを明示
Settings.llm = granite_llm

#  Embedding/Index 構築 
# 研究目的の簡潔なコメント: 日本語向け埋め込みを採用しPDFをベクトル化する.
Settings.embed_model = HuggingFaceEmbedding(
    model_name="pkshatech/GLuCoSE-base-ja"
)

# PDF_PATH = os.getenv("RAG_PDF_PATH", "./docs/housetomato.pdf")
PDF_PATH = os.getenv("RAG_PDF_PATH", "./docs/example2.pdf")
loader = PyMuPDFReader()
documents = loader.load(file_path=PDF_PATH)

splitter = SentenceSplitter(chunk_size=512)
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[splitter],
    embed_model=Settings.embed_model,
)

#  Retriever 構築（ハイブリッド検索） 
# 研究目的の簡潔なコメント: ベクトルとBM25を融合したQueryFusionRetrieverを使用する.
vector_retriever = index.as_retriever(similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,
    similarity_top_k=2,
)

QUERY_GEN_PROMPT = (
    "あなたは，1つの入力クエリに基づいて複数の検索クエリを生成する有能なアシスタントです.\n"
    "{num_queries}個の検索クエリを，1行につき1つずつ生成してください.\n"
    "以下のクエリに関連する検索クエリを生成してください:\n\n"
    "クエリ: {query}\n"
    "検索クエリ:\n"
)

fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=4,
    num_queries=4,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=False,
    query_gen_prompt=QUERY_GEN_PROMPT,
    llm=granite_llm,
)

#  QueryEngine を2系統作成 -
# 各エンジンは同一コーパスを検索するが，用いるLLMとサブクエリ生成器を分離する．
from llama_index.core.response_synthesizers import get_response_synthesizer

# サブクエリ生成用にLLMを明示した Retriever（Granite用／Llama用）
fusion_retriever_granite = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=4,
    num_queries=4,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=False,
    query_gen_prompt=QUERY_GEN_PROMPT,
    llm=granite_llm,
)

fusion_retriever_llama = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=4,
    num_queries=4,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=False,
    query_gen_prompt=QUERY_GEN_PROMPT,
    llm=llama_llm,
)

# 応答生成器（Response Synthesizer）をモデルごとに用意
granite_synth = get_response_synthesizer(response_mode="compact", llm=granite_llm)
llama_synth = get_response_synthesizer(response_mode="compact", llm=llama_llm)

# RetrieverQueryEngine は llm 引数を持たないため，response_synthesizer を渡す
# （Settings.llm に依存しないように明示する）
from llama_index.core.query_engine import RetrieverQueryEngine

granite_engine = RetrieverQueryEngine(
    retriever=fusion_retriever_granite,
    response_synthesizer=granite_synth,
)

llama_engine = RetrieverQueryEngine(
    retriever=fusion_retriever_llama,
    response_synthesizer=llama_synth,
)

#  推論関数 -
# 研究目的の簡潔なコメント: 同一入力に対する2モデルの応答を返却する.

def compare_models(prompt: str) -> Tuple[str, str]:
    try:
        g_resp = granite_engine.query(prompt)
        l_resp = llama_engine.query(prompt)
        g_text = getattr(g_resp, "response", str(g_resp))
        l_text = getattr(l_resp, "response", str(l_resp))
        return g_text, l_text
    except Exception as e:
        err = f"エラーが発生しました: {e}"
        return err, err

#  Gradio UI 
# 研究目的の簡潔なコメント: 単一入力で2出力を並列表示するUIを提供する.
with gr.Blocks(title="Granite vs Llama — RAG出力比較") as demo:
    gr.Markdown("""
    # Granite vs Llama — RAG出力比較
    同一の質問を2つのLLMに投げて結果を比較します．PDF: `{}`
    """.format(PDF_PATH))

    with gr.Row():
        inp = gr.Textbox(label="質問", placeholder="例: トマトの栽培で重要な管理ポイントは?", lines=3)
    with gr.Row():
        out_g = gr.Textbox(label="Granite の回答", lines=12)
        out_l = gr.Textbox(label="Llama の回答", lines=12)
    with gr.Row():
        btn = gr.Button("比較する")

    examples = gr.Examples(
        examples=[
            ["トマトを家庭で育てるにはどうすればよいですか？"],
            ["トマトの栽培に最適な気候や土壌条件は何ですか？"],
        ],
        inputs=[inp],
    )

    btn.click(fn=compare_models, inputs=inp, outputs=[out_g, out_l])

if __name__ == "__main__":
    demo.launch(inline=False, share=False)

