import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import PyPDF2
import tempfile

MODEL_NAME = "microsoft/BioGPT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def read_file(file):
    if file.name.endswith(".pdf"):
        pdf = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf.pages])
    else:
        text = file.read().decode("utf-8")
    return text.strip()

def generate_response(prompt, max_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_tokens,
            num_beams=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

def generate_summary_prompt(text):
    return f"Summarize the following medical note:\n{text}\nSummary:"

def generate_qa_prompt(text, question):
    return f"Medical record: {text}\nQuestion: {question}\nAnswer:"

st.set_page_config(page_title="BioGPT Medical Analyzer", layout="wide")

st.title("🧠 BioGPT Medical Report Summarizer")
st.markdown("Upload a medical text or PDF file, and generate a summary or answer specific questions.")

uploaded_file = st.file_uploader("📄 Upload Medical File (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    text = read_file(uploaded_file)
    st.subheader("📃 Extracted Text:")
    st.text_area("File Content", text, height=200)

    if st.button("📝 Summarize"):
        with st.spinner("Generating summary..."):
            prompt = generate_summary_prompt(text)
            summary = generate_response(prompt)
            st.success("Summary Generated:")
            st.write(summary)

    st.subheader("❓ Ask a Medical Question")
    question = st.text_input("Enter your question about the file content")

    if st.button("💬 Get Answer") and question:
        with st.spinner("Generating answer..."):
            prompt = generate_qa_prompt(text, question)
            answer = generate_response(prompt)
            st.success("Answer:")
            st.write(answer)
