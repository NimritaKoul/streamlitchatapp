#My Streamlit chat app to chat with mistral7b
import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
from io import BytesIO
import docx


def extract_text_from_pdf(uploaded_file):
    text = ""
    with BytesIO(uploaded_file.read()) as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


def extract_text_from_docx(uploaded_file):
    text = ""
    doc = docx.Document(uploaded_file)
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text





st.set_page_config(page_title="Extractive Question Answering")
st.title("Ask questions from your documents.")

"""
[![LinkedIn](https://img.shields.io/badge/LinkedIn-NimritaKoul-blue?style=social&logo=linkedin)](https://www.linkedin.com/in/nimritakoul/)
"""

st.sidebar.header("This app answers your questions from the documents you choose.")

bullet_list = """
* You can upload a PDF, a Docx or a text file and then ask any questions from the text contained in the file.
* The language model this app uses is deepset/tinyroberta-squad2 from HuggingFace.
* It is an extractive QA model, i.e., it extracts answers from the context or the text you provide to it.
* If it can't find the answer in your document, it will return an empty string.
* It is not a generative QA model.
"""


# Display long text in the sidebar
st.sidebar.markdown(bullet_list)



url = "https://huggingface.co/deepset/tinyroberta-squad2"
link_text = "Link to TinyRobera-Squad2 at HuggingFace"
st.sidebar.markdown(f"[{link_text}]({url})")

st.sidebar.subheader("**Choose a file type you wish to work with:**")
option = st.sidebar.radio("", ["Ask from a PDF File", "Ask from a Word File", "Ask from a text file"])

@st.cache_resource(show_spinner=False)
def question_model():
    model_name = "deepset/tinyroberta-squad2"
    question_answerer = pipeline(model=model_name, tokenizer=model_name, task="question-answering")
    return question_answerer

if option == "Ask from a PDF File":
    st.markdown("<h2 style='text-align: center; color:grey;'>Ask from a PDF File</h2>", unsafe_allow_html=True)
    
    st.text("Please upload a PDF file upto 3 MB in Size")
    uploaded_file = st.file_uploader("Choose a PDF file to upload", type=["pdf"])
    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        context = st.text_area("", value=pdf_text, height=330)
        question = st.text_input(label="Enter your question")
        button = st.button("Get answer")
        if button:
            with st.spinner(text="Loading QA model..."):
                question_answerer = question_model()
            with st.spinner(text="Getting answer..."):
                answer = question_answerer(context=context, question=question)
                answer = answer["answer"]
                st.markdown("**Answer:**")
                st.text(answer)

        
elif option == "Ask from a Word File":
    st.markdown("<h2 style='text-align: center; color:grey;'>Ask from a Word(Docx) File</h2>", unsafe_allow_html=True)
    
    st.text("Please upload a Docx file upto 3 MB in Size")
    uploaded_file = st.file_uploader("Choose a .docx file to upload", type=["docx"])
    if uploaded_file is not None:
        docx_text = extract_text_from_docx(uploaded_file)
        context = st.text_area("", value=docx_text, height=330)
        question = st.text_input(label="Enter your question")
        button = st.button("Get answer")
        if button:
            with st.spinner(text="Loading QA model..."):
                question_answerer = question_model()
            with st.spinner(text="Getting answer..."):
                answer = question_answerer(context=context, question=question)
                answer = answer["answer"]
                st.markdown("**Answer:**")
                st.text(answer)
 
elif option == "Ask from a text file":
    st.markdown("<h2 style='text-align: center; color:grey;'>Ask from a .txt File</h2>", unsafe_allow_html=True)
    
    st.text("Please upload a .txt file upto 3 MB in Size")
    uploaded_file = st.file_uploader("Choose a .txt file to upload", type=["txt"])
    if uploaded_file is not None:
        raw_text = str(uploaded_file.read(),"utf-8")
        context = st.text_area("", value=raw_text, height=330)
        question = st.text_input(label="Enter your question")
        button = st.button("Get answer")
        if button:
            with st.spinner(text="Loading QA model..."):
                question_answerer = question_model()
            with st.spinner(text="Getting answer..."):
                answer = question_answerer(context=context, question=question)
                answer = answer["answer"]
                st.markdown("**Answer:**")
                st.text(answer)
  