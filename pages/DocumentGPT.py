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
import os

st.set_page_config(page_title="DocumentGPT", page_icon="📃",)
with st.sidebar:
    openaikey = None
    openaikey = st.text_input("Your OpenAI API key: ", type="password")
    os.environ["OPENAI_API_KEY"] = openaikey


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    # LLM 실행 시작 시, 비어있는 출력 창을 생성
    # 스트리밍 메시지를 표시할 공간 생성
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    # LLM 실행 완료 후, 최종 메시지를 저장
    # AI 응답을 저장
    def on_llm_end(self, *args, **kwargs): 
        save_message(self.message, "ai") 

    # # AI 응답을 스트리밍 방식으로 표시(실시간으로 한 글자씩 출력)
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token # 새로운 토큰을 메시지에 추가
        self.message_box.markdown(self.message) # 메시지 업데이트 및 출력

llm = ChatOpenAI(temperature=0.5, streaming=True, callbacks=[ChatCallbackHandler()])


st.title("Document GPT")
st.markdown("""
Welcome!
            
Use this chatbot to ask questions to an AI about yout files!!
            
Upload your files on the sidebar
""")


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings() # 문서를 벡터화
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir) # 임베딩을 캐싱 (속도 향상)

    vectorstore = FAISS.from_documents(docs, cached_embeddings) # FAISS(빠른 유사도 검색 라이브러리)를 활용한 벡터스토어 구축
    retriever = vectorstore.as_retriever() # 검색 기능 활성화
    return retriever
    

# 채팅을 저장하여 히스토리를 유지
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


# 기본적으로 메시지를 st.session_state["messages"]에 저장할지 여부를 결정
# 메시지를 화면에 출력
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

# 저장된 모든 메시지를 화면에 다시 출력
# save=False -> 이전에 저장된 메시지를 단순히 화면에 다시 보여주는 것
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", """
    All answer in Korean.
    Answer the question using ONLY the following context.
    If you don't know the answer just say you dont't know.
    Don't make anything up.
    Context: {context}
    """),
    ("human", "{question}")
])


with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"])


if file:
    retriever = embed_file(file)


    send_message("궁금한 것을 물어보세요!", "ai", save=False)
    paint_history()


    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = ({"context":retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()} | prompt | llm)
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []

#  session_state 초기화
# "messages" 키가 session_state에 없으면, 빈 리스트를 생성해서 저장 -> 이전 메시지를 저장하고 유지
# if "messages" not in st.session_state:
#     st.session_state["message"] = []