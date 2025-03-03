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
from langchain.schema import BaseOutputParser
import json
import os


st.set_page_config(page_title="QuizGPT", page_icon="🔥",)
st.title("Document GPT")


llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-0125", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)
output_parser = JsonOutputParser()


questions_prompt = ChatPromptTemplate.from_messages([(
    "system",
    """
        당신은 교사 역할을 하는 도움이 되는 조수입니다.

        다음 컨텍스트를 기반으로 오직 해당 내용만 참고하여 사용자의 지식을 테스트할 10개의 질문을 만들어 주세요.

        각 질문에는 4개의 선택지가 있어야 하며, 이 중 하나만 정답이고 나머지 세 개는 오답이어야 합니다.

        정답에는 (o) 표시를 해주세요.

        질문 예시:
        질문: 바다의 색은 무엇인가요?
        답변: 빨강 | 노랑 | 초록 | 파랑(o)

        질문: 조지아(Georgia)의 수도는 어디인가요?
        답변: 바쿠 | 트빌리시(o) | 마닐라 | 베이루트

        질문: 영화 아바타는 언제 개봉되었나요?
        답변: 2007 | 2001 | 2009(o) | 1998

        질문: 줄리어스 시저는 누구인가요?
        답변: 로마 황제(o) | 화가 | 배우 | 모델

        이제 당신의 차례입니다!

        Context: {context}
    """
)])
questions_chain = {"context": format_docs} | questions_prompt | llm


formatting_prompt = ChatPromptTemplate.from_messages([(
    "system",
    """
        당신은 강력한 포맷팅 알고리즘입니다.

        당신의 역할은 시험 문제를 JSON 형식으로 변환하는 것입니다.
        문항 중에서 (o) 표시가 있는 선택지가 정답입니다.

        입력 예시:
        Question: 바다의 색은 무엇인가요?
        Answers: 빨강 | 노랑 | 초록 | 파랑(o)

        Question: 조지아(Georgia)의 수도는 어디인가요?
        Answers: 바쿠 | 트빌리시(o) | 마닐라 | 베이루트

        Question: 영화 아바타는 언제 개봉되었나요?
        Answers: 2007 | 2001 | 2009(o) | 1998

        Question: 줄리어스 시저는 누구인가요?
        Answers: 로마 황제(o) | 화가 | 배우 | 모델

        Example Output:

        ```json
        {{ "questions": [
            {{
                "question": "바다의 색은 무엇인가요?",
                "answers": [
                    {{
                        "answer": "빨강",
                        "correct": false
                    }},
                    {{
                        "answer": "노랑",
                        "correct": false
                    }},
                    {{
                        "answer": "초록",
                        "correct": false
                    }},
                    {{
                        "answer": "파랑",
                        "correct": true
                    }}
                ]
            }},
                {{
            "question": "조지아(Georgia)의 수도는 어디인가요?",
            "answers": [
                {{
                    "answer": "바쿠",
                    "correct": false
                }},
                {{
                    "answer": "트빌리시",
                    "correct": true
                }},
                {{
                    "answer": "마닐라",
                    "correct": false
                }},
                {{
                    "answer": "베이루트",
                    "correct": false
                }}
            ]
        }},
            {{
                "question": "영화 아바타는 언제 개봉되었나요?",
                "answers": [
                {{
                    "answer": "2007",
                    "correct": false
                }},
                {{
                    "answer": "2001",
                    "correct": false
                }},
                {{
                    "answer": "2009",
                    "correct": true
                }},
                {{
                    "answer": "1998",
                    "correct": false
                }}
            ]
        }},
            {{
                "question": "줄리어스 시저는 누구인가요?",
                "answers": [
                {{
                    "answer": "로마 황제",
                    "correct": true
                }},
                {{
                    "answer": "화가",
                    "correct": false
                }},
                {{
                    "answer": "배우",
                    "correct": false
                }},
                {{
                    "answer": "모델",
                    "correct": false
                }}
            ]
        }}
    ]
    }}
    ```
    이제 당신의 차례입니다!
    Questions: {context}
    """
)]) # {{}} 중괄호 두 개 -> 해당 부분은 format 하지 않기를 바랄 때 사용
formatting_chain = formatting_prompt | llm



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


# Streamlit은 @st.cache_data를 사용하여 함수의 입력값을 해시(hash)하여 캐싱
# 이때, 리스트(list), 딕셔너리(dict) 같은 변형 가능한 객체(mutable objects)는 해싱할 수 없기 때문에 오류가 발생
# 함수의 인자에 언더스코어(_)를 붙이면 해당 인자를 해싱하지 않고 그대로 유지
@st.cache_data(show_spinner="퀴즈 만드느 중...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="위키피디아 검색 중...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term, lang="ko")
    return docs


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
            docs = wiki_search(topic)


if not docs:
    st.markdown(
    """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """)
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio("선택지", [answer["answer"] for answer in question["answers"]], index=None,)

            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
                
        button = st.form_submit_button()
