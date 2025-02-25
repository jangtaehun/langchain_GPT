from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.retrievers import WikipediaRetriever
from langchain.callbacks import StreamingStdOutCallbackHandler
import os


st.set_page_config(page_title="QuizGPT", page_icon="🔥",)
st.title("Document GPT")


llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-0125", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return  docs


with st.sidebar:
    docs = None

    choice = st.selectbox("Choose what you want to use.", ("File", "Wikipedia Article"))

    if choice == "File":
        file = st.file_uploader("Upload a .docs, .txt or .pdf file", type=["pdf", "txt", "docs"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            retriever = WikipediaRetriever(top_k_results=2)
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic, lang="ko")


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    당신은 교사 역할을 하는 유용한 AI 어시스턴트입니다.
    
    아래 제공된 **컨텍스트**를 기반으로만 사용자의 지식을 테스트할 10개의 질문을 만들어 주세요.
    
    각 질문에는 **4개의 보기**가 포함되어야 하며, 그중 하나만 정답이고 나머지 3개는 오답이어야 합니다.
    
    정답에는 (o)를 표시해 주세요.
    
    **질문 예시:**
    
    질문: 바다가 가진 색깔은 무엇인가요?
    보기: 빨강|노랑|초록|파랑(o)
    
    질문: 조지아(Georgia)의 수도는 어디인가요?
    보기: 바쿠|트빌리시(o)|마닐라|베이루트
    
    질문: 영화 "아바타"는 언제 개봉했나요?
    보기: 2007|2001|2009(o)|1998
    
    질문: 줄리어스 시저는 누구인가요?
    보기: 로마 황제(o)|화가|배우|모델
    
    이제 당신이 직접 만들어 보세요!
    
    **컨텍스트:** {context}
""",
            )
        ]
    )


    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)
    chain = {"context":format_docs} | prompt | llm
    start = st.button("Generate Quiz")
    if start:
        chain.invoke(docs)