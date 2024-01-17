import streamlit as st
from transformers import pipeline

model_name = "deepset/roberta-base-squad2"
QA_model = pipeline('question-answering', model=model_name, tokenizer=model_name)
def main():
    st.title("Ответ на вопрос по тексту при помощи модели Roberta-base :blue[(Hugging Face Model)] :hugging_face:")
    st.image("Leonardo_Diffusion_XL_Question_Answering_with_Hugging_Face.jpg", use_column_width=True)
    input_text = st.text_input("Скопируйте сюда Ваш текст", "")
    input_question = st.text_input("Введите Ваш вопрос", "")
    if st.button("Получить ответ"):
        answer = QA_model({
            'question': input_question,
            'context': input_text,
        })
        st.write("Ответ:", answer['answer'])

if __name__ == '__main__':
    main()
