import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

@st.cache_resource  # 👈 Add the caching decorator
def load_model():    
    return pipeline('zero-shot-classification', model="facebook/bart-large-mnli")

st.title('Text Classification with BART!')
pipe = load_model()

text = st.text_input('Tell me something about yourself')

st.text(text)
if text != '':
    resp = pipe(text, candidate_labels=["sounds sad", "big news", "curious fact"],)
    st.write(resp)
