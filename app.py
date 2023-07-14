import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os


st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📚"
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

with st.sidebar:
    st.title("Chat with PDF")
    st.markdown('''
    ## About 
    This app powered chat bot build using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/docs/get_started/introduction.html)
    - [OpanAI](https://openai.com/)
                
    ''')
    add_vertical_space(5)
    st.write("Made by [Anant Tater](https://github.com/ananttater)")

load_dotenv()

def main():
    st.markdown("<h1 style='text-align: center; color: white;'>Chat with PDF 💬</h1>", unsafe_allow_html=True)
    
    #upload the pdf file
    pdf = st.file_uploader("Upload your PDF file her..  ", type=['pdf'])

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)

        text = " "
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        # st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # embedding = OpenAIEmbeddings()
        # VectoreStore = FAISS

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

        # st.write(chunks)
    
    

if __name__ == '__main__':
    main()