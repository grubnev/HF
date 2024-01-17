import streamlit as st
from transformers import pipeline

model_name = "deepset/roberta-base-squad2"
QA_model = pipeline('question-answering', model=model_name, tokenizer=model_name)
def main():
    st.title("Question Answering with Hugging Face Model")
    input_text = st.text_input("Enter your text", "")
    input_question = st.text_input("Enter your question", "")
    if st.button("Answer"):
        answer = QA_model({
            'question': input_question,
            'context': input_text,
        })
        st.write('Answer:', answer['answer'])

if __name__ == '__main__':
    main()
