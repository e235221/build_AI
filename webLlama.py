# --- 追加パッケージ ---
# pip install duckduckgo-search readability-lxml beautifulsoup4 lxml requests

import os
import re
import gradio as gr
import requests
from bs4 import BeautifulSoup
from readability import Document
from duckduckgo_search import DDGS

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

os.environ["WATSONX_APIKEY"] = "0i-_-6pigerNnnRaU8_oiybRZz_UxMQuBHpE_copxSdw"
os.environ["WATSONX_PROJECT_ID"] = "b596c884-f867-4771-afcc-f9fd10dae1a4"

def initialize_model():
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 50,
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 512,
        GenTextParamsMetaNames.TEMPERATURE: 0.7,
        GenTextParamsMetaNames.TOP_K: 50,
        GenTextParamsMetaNames.TOP_P: 0.9,
    }
    model = Model(
        model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        params=parameters,
        credentials={"apikey": os.getenv("WATSONX_APIKEY"),
                     "url": "https://us-south.ml.cloud.ibm.com"},
        project_id=os.getenv("WATSONX_PROJECT_ID"),
    )
    return model

model = initialize_model()

# ========== ここから “Web-LLM Assistant” 的な検索+スクレイプ層 ==========

UA = {"User-Agent": "Mozilla/5.0"}

def url_alive(url: str, timeout: int = 10) -> bool:
    try:
        # HEADがブロックされるサイトもあるのでGETで確認（サイズを抑える）
        r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True, stream=True)
        ct = r.headers.get("Content-Type","")
        return (r.status_code == 200) and ("text/html" in ct or "application/xhtml" in ct)
    except Exception:
        return False
    
def ddg_search(query: str, max_results: int = 10, timelimit: str | None = None):
    """
    DuckDuckGoのテキスト検索。
    timelimit: 'd'|'w'|'m'|'y'（日/週/月/年） 例: timelimit='w'
    """
    results = DDGS().text(
        query, region="jp-jp", safesearch="moderate",
        timelimit=timelimit, max_results=max_results
    )
    # 結果は {"title","href","body"} のリスト（duckduckgo-search仕様）:contentReference[oaicite:2]{index=2}
    return results or []

def fetch_article(url: str, timeout: int = 12) -> tuple[str, str]:
    """
    URLから本文抽出（readability-lxmlでメイン本文抽出→テキスト化）。
    """
    try:
        resp = requests.get(url, headers=UA, timeout=timeout)
        resp.raise_for_status()
        doc = Document(resp.text)
        title = doc.title() or ""
        html = doc.summary()
        text = BeautifulSoup(html, "lxml").get_text("\n", strip=True)
        # ノイズ削り
        text = re.sub(r"\n{3,}", "\n\n", text)
        return title, text
    except Exception:
        return "", ""

def gather_context(query: str, timelimit: str | None = None, max_pages: int = 3):
    """
    DuckDuckGoで検索 → 到達性のあるURLだけに絞る → 公式ドメインを優先 →
    上位から本文抽出し、最大 max_pages 件を文脈として返す。
    """
    hits = ddg_search(query, max_results=15, timelimit=timelimit)

    # 公式ドメイン優先のスコア（必要に応じて調整）
    def _official_score(href: str) -> int:
        href = href or ""
        return (
            (3 if "pref.okinawa.jp" in href else 0) +
            (3 if "city.nago.okinawa.jp" in href else 0) +
            (2 if ".go.jp" in href else 0) +
            (2 if ".lg.jp" in href else 0) +
            (1 if href.endswith(".jp") or ".jp/" in href else 0)
        )

    # 1) URLを取り出し → 2) 到達性チェック → 3) 公式スコアで降順ソート
    candidates = []
    for h in hits:
        url = h.get("href") or h.get("url")
        if not url:
            continue
        if not url_alive(url):  # ★ 開けないURLは弾く
            continue
        candidates.append((h.get("title", ""), url, _official_score(url)))

    candidates.sort(key=lambda x: x[2], reverse=True)

    sources = []
    context_chunks = []
    seen_domains = set()

    for title0, url, _sc in candidates:
        # 同一ドメインを取り過ぎないように制御
        domain = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        if domain in seen_domains:
            continue

        title, text = fetch_article(url)
        if len(text) < 500:  # 短すぎる/抜け時はスキップ
            continue

        idx = len(sources) + 1
        sources.append((idx, title or title0, url))
        # 各記事は長すぎないように冒頭 ~3000 文字に制限
        context_chunks.append(f"[{idx}] {title or title0}\n{text[:3000]}")
        seen_domains.add(domain)

        if len(sources) >= max_pages:
            break

    return sources, "\n\n---\n\n".join(context_chunks)


def answer_with_web(query: str, timelimit: str | None = None):
    """
    文脈(検索+スクレイプ)をLLMに渡して最終回答を生成。
    """
    sources, context = gather_context(query, timelimit=timelimit, max_pages=3)
    if not sources:
        # もし何も取れなかったら通常回答にフォールバック
        return model.generate_text(prompt=query)

    # LLMへのプロンプト（単一prompt方式）
    src_lines = "\n".join([f"[{i}] {t} — {u}" for i, t, u in sources])
    prompt = f"""あなたは事実重視のリサーチアシスタントです。以下の資料だけを根拠に日本語で回答してください。
- 引用は [番号] で示し、最後に参照URLを列挙してください。
- 不確実な点はその旨を述べ、推測は避けてください。

# ユーザーの質問
{query}

# 参考資料（抜粋テキスト）
{context}

# 出力フォーマット
1) 要点の結論
2) 補足（根拠： [1][2] のように番号で）
3) 参照URL一覧（番号→URL）

回答："""

    answer = model.generate_text(prompt=prompt)
    return f"{answer}\n\n参照URL:\n" + "\n".join([f"[{i}] {u}" for i,_,u in sources])

# ========== Gradioハンドラ ==========

def chat_function(message, history):
    try:
        # "/web " で始まる入力はWebアシスト（timelimitは必要に応じて 'd','w','m','y' で）
        if message.startswith("/web "):
            q = message[len("/web "):].strip()
            # 例: 直近1週間に絞るなら timelimit='w'
            return answer_with_web(q, timelimit=None)
        else:
            return model.generate_text(prompt=message)
    except Exception as e:
        return f"Error occurred: {str(e)}"

demo = gr.ChatInterface(
    fn=chat_function,
    title="Llamaチャット（/web で検索対応）",
    description="先頭に /web を付けるとWeb検索＋スクレイピングで根拠URL付き回答（DuckDuckGo＋readability）。",
    theme="soft",
    examples=[
        "/web 直近の日本の物価動向をざっくり3点で。根拠URL付きで",
        "/web 以下のプロフィールの就農者が受給できる補助金5つをURL付きで提案してください ①所在地：沖縄県・名護市②経営状況：農業ベンチャー企業・中規模（年産 120 トン）③作目：パイナップル ④目的：ジュース開発",
        "通常の雑談は /web なしでOK",
    ],
)

demo.launch(share=False)
