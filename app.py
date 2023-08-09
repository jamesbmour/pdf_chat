import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import InstructorEmbedding
# load environment variables
load_dotenv()
# get pdf text method
def get_pdf_text(pdf_file):
    """
    Get raw text from pdf file
    :param pdf_file: pdf file
    :return: raw text
    """
    text = ""
    for pdf in pdf_file:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# get text chunks method
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Get text chunks from raw text to avoid missing information

    :param text: raw text from pdf
    :param chunk_size: size of each chunk of text
    :param chunk_overlap: overlap between chunks to avoid missing information
    :return: list of text chunks from raw text
    """
    text_chunks = []
    position = 0
    # Iterate over the text until the entire text has been processed
    while position < len(text):
        # Calculate the starting index for the current chunk,
        # ensuring it doesn't go below 0
        start_index = max(0, position - chunk_overlap)
        # Calculate the ending index for the current chunk
        end_index = position + chunk_size
        # Extract the current chunk of text using slicing
        chunk = text[start_index:end_index]
        # Add the extracted chunk to the list of text chunks
        text_chunks.append(chunk)
        # Update the position for the next iteration, accounting for overlap
        position = end_index - chunk_overlap
    return text_chunks


# get vector store method
def get_vectorstore(text_chunks):
    """
    Get vector store from text chunks using language model
    :param text_chunks: list of text chunks
    :return:  vector store
    """

    # get openai api key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # get embeddings
    print(openai_api_key)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


# get conversation chain method
def get_conversation_chain(vectorstore):
    """
    Get conversation chain from vector store

    :param vectorstore: vector store
    :return: conversation chain
    """
    model_kwargs = {"temperature": 0.5, "max_length": 4096}
    # Initialize a language model for chat-based interaction (LLM)
    llm = ChatOpenAI(tempatature=0.5)

    # Alternatively, you can use a different language model, like Hugging Face's model
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":4096}, api_key=os.getenv("HUGGINGFACE_API_TOKEN"))

    # Initialize a memory buffer to store conversation history
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    # Create a conversation chain for conversational retrieval
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Language model for generating responses
        retriever=vectorstore.as_retriever(),  # Text vector retriever for context matching
        memory=memory  # Memory buffer to store conversation history
    )

    return conversation_chain


# get handler user input method
def handle_userinput(user_question):
    """
    Get handler user input from conversation chain
    :param user_question:  conversation chain
    :return:  handler user input
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)






def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


    # init sidebar
    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs and click process", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing PDFs"):
                print("Processing PDFs")
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)
                print(raw_text)
                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                print(f'Number of text chunks: {len(text_chunks)}')
                # get vector store
                print("Creating vector store")
                vector_store = get_vectorstore(text_chunks)
                print("Vector store created")
                # print(f'Number of vectors: {len(vector_store)}')
                # get conversation chain
                print("Creating conversation chain")
                # links variable to session state so it can be used in other functions and not recreated
                st.session_state.conversation = get_conversation_chain(vector_store)
                print("Conversation chain created")

    # get handler user input








if __name__ == '__main__':
    main()