import os
import gradio as gr
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
        GenTextParamsMetaNames.TOP_P: 0.9
    }

    model = Model(
        model_id="openai/gpt-oss-120b",
        params=parameters,
        credentials={
            "apikey": os.getenv("WATSONX_APIKEY"),
            "url": "https://us-south.ml.cloud.ibm.com"
        },
        project_id=os.getenv("WATSONX_PROJECT_ID")
    )
    
    return model

model = initialize_model()

def chat_function(message, history):
    try:
        response = model.generate_text(prompt=message)
        return response
    except Exception as e:
        return f"Error occurred: {str(e)}"

demo = gr.ChatInterface(
    fn=chat_function,
    title="GPT4oチャット",
    description="WatsonX経由でIBM GPT4oモデルと直接チャット",
    theme="soft",
    examples=[
        "こんにちは、お元気ですか？",
        "人工知能とは何ですか？",
        "ロボットについての短い物語を教えてください。",
        "量子コンピューティングを簡単な言葉で説明してください。"
    ]
)

demo.launch(share=False)
