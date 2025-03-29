from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    pipeline, AutoModelForSequenceClassification
)
import torch

import os
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Load Models
# Summarization model
summary_model_id = "machinelearningzuu/ptsd-summarization"
summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_id)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_id)
summarizer = pipeline("summarization", model=summary_model, tokenizer=summary_tokenizer)

# Sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")

# Emotion classifier
clf_model_id = "nateraw/bert-base-uncased-emotion"
clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_id)
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_model_id)
clf_model.eval()
clf_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# DeepSeek
deepseek_model_id = "deepseek-ai/deepseek-llm-7b-chat"
deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_id, token=hf_token)
deepseek_model = AutoModelForCausalLM.from_pretrained(
    deepseek_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token
)
deepseek_tokenizer.pad_token = deepseek_tokenizer.eos_token
torch.backends.cuda.matmul.allow_tf32 = True

# FastAPI App
app = FastAPI()

class PTSDInput(BaseModel):
    patient_text: str

@app.post("/suggest")
def analyze_ptsd(data: PTSDInput):
    text = data.patient_text

    # Summary
    summary_output = summarizer(text, max_length=100, min_length=10, do_sample=False)
    summary = summary_output[0]['summary_text']

    # Sentiment
    sentiment_output = sentiment_analyzer(text)[0]
    sentiment = f"{sentiment_output['label']} ({sentiment_output['score']:.2f})"

    # Emotion classification
    inputs = clf_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = clf_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_idx = torch.argmax(probs).item()
        emotion = f"{clf_labels[top_idx].capitalize()} ({probs[0][top_idx].item():.2f})"

    # Coping Suggestion (DeepSeek)
    prompt = (
        f"Patient summary: {summary}\n"
        f"Based on this, provide 3 specific coping suggestions for PTSD symptoms:\n"
        f"1."
    )
    deep_inputs = deepseek_tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(deepseek_model.device)

    with torch.inference_mode():
        deep_outputs = deepseek_model.generate(
            input_ids=deep_inputs["input_ids"],
            attention_mask=deep_inputs["attention_mask"],
            max_new_tokens=100,
            do_sample=False,
            temperature=0.7,
            eos_token_id=deepseek_tokenizer.eos_token_id,
            pad_token_id=deepseek_tokenizer.pad_token_id
        )
    generated = deepseek_tokenizer.decode(deep_outputs[0], skip_special_tokens=True)
    suggestion = "1. " + generated.split("1.", 1)[-1].strip()

    return {
        "summary": summary,
        "sentiment": sentiment,
        "mental_health_indicator": emotion,
        "coping_suggestions": suggestion
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
