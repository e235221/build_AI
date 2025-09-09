import os
import gradio as gr
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

os.environ["WATSONX_APIKEY"] = "0i-_-6pigerNnnRaU8_oiybRZz_UxMQuBHpE_copxSdw"
os.environ["WATSONX_PROJECT_ID"] = "b596c884-f867-4771-afcc-f9fd10dae1a4"

def initialize_model():
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 50,
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 1024,
        GenTextParamsMetaNames.TEMPERATURE: 0.7,
        GenTextParamsMetaNames.TOP_K: 50,
        GenTextParamsMetaNames.TOP_P: 0.9,
    }

    try:
        model = ModelInference(
            model_id="openai/gpt-oss-120b",
            params=parameters,
            credentials={
                "apikey": os.getenv("WATSONX_APIKEY"),
                "url": "https://us-south.ml.cloud.ibm.com"
            },
            project_id=os.getenv("WATSONX_PROJECT_ID")
        )
        return model
    except Exception as e:
        raise

model = initialize_model()

def chat_function(message, history):
    try:
        messages = [
            {"role": "system", "content": "Reasoning: medium\nあなたは親切なアシスタントです。"},
        ]
        messages.append({"role": "user", "content": message})
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        
        response = model.chat(messages=messages)
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error occurred: {str(e)}"

demo = gr.ChatInterface(
    fn=chat_function,
    title="GPT-4oチャット",
    description="gpt-4oチャット",
    theme="soft",
    examples=[
        "こんにちは、お元気ですか？",
        "人工知能とは何ですか？",
        "ロボットについての短い物語を教えてください。",
        "量子コンピューティングを簡単な言葉で説明してください。"
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)
