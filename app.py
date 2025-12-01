# app_gradio.py
# Gradio UI for PhishGuard URL classifier

import re
import math
import tldextract
import pandas as pd
import joblib
import gradio as gr


MODEL_PATHS = {
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}


label_map = {
    0: "phishing",
    1: "defacement",
    2: "malware",
    3: "benign"
}

# ---------- Utilities ----------
def entropy(s):
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log(p, 2) for p in probs)

def extract_features(url: str):
    if url is None:
        url = ""
    url = str(url).strip()
    ext = tldextract.extract(url)
    domain = ext.domain or ""
    subdomain = ext.subdomain or ""

    features = {}
    features["url_length"] = len(url)
    features["domain_length"] = len(domain)
    features["subdomain_length"] = len(subdomain)
    features["path_length"] = len(url.split("/")[-1]) if "/" in url else 0

    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_letters"] = sum(c.isalpha() for c in url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_slashes"] = url.count("/")
    features["num_special_chars"] = sum(not c.isalnum() for c in url)

    features["https"] = 1 if url.lower().startswith("https") else 0
    features["has_ip"] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0

    suspicious_words = ["login", "verify", "bank", "secure", "update", "free", "confirm", "paypal"]
    features["sus_word_present"] = 1 if any(w in url.lower() for w in suspicious_words) else 0

    features["digit_ratio"] = features["num_digits"] / (features["url_length"] + 1)
    features["special_ratio"] = features["num_special_chars"] / (features["url_length"] + 1)
    features["entropy"] = entropy(url)

    return features


# ---------- Dynamic Model Loader + Prediction ----------
def predict_single(url: str, model_name: str):
    if not url or url.strip() == "":
        return {"label": "error", "message": "Empty URL provided."}

    # Load selected model
    try:
        model_path = MODEL_PATHS[model_name]
        model = joblib.load(model_path)
    except Exception as e:
        return {"label": "error", "message": f"Failed to load model: {e}"}

    # Extract features
    features = extract_features(url)
    df = pd.DataFrame([features])

    try:
        pred_id = int(model.predict(df)[0])
        label = label_map.get(pred_id, f"unknown({pred_id})")
        proba = None

        if hasattr(model, "predict_proba"):
            proba_arr = model.predict_proba(df)[0]
            proba = float(proba_arr[pred_id])

        return {
            "label": label,
            "probability": proba,
            "features": features
        }

    except Exception as e:
        return {"label": "error", "message": str(e)}


# ---------- Gradio UI ----------
with gr.Blocks() as demo:

    gr.Markdown("# 🛡️ PhishGuard — URL Classifier")
    gr.Markdown("Choose a model (Random Forest / XGBoost) and enter a URL.")

    with gr.Row():
        url_in = gr.Textbox(label="URL", placeholder="https://example.com/login")
        model_selector = gr.Dropdown(
            choices=["Random Forest", "XGBoost"], 
            label="Choose Model",
            value="XGBoost"
        )
        predict_btn = gr.Button("Predict")

    output_label = gr.Textbox(label="Prediction")
    output_prob = gr.Textbox(label="Probability (if available)")
    output_features = gr.JSON(label="Extracted Features (debug)")

    def run(url, model_name):
        r = predict_single(url, model_name)
        if r.get("label") == "error":
            return r.get("message"), "", {}
        return r["label"], str(r.get("probability")), r.get("features")

    predict_btn.click(run, 
                      inputs=[url_in, model_selector], 
                      outputs=[output_label, output_prob, output_features])


# ---------- Launch ----------
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

