python -m venv ./env

# mac
> - bash/zsh: source <venv>/bin/activate
> - PowerShell: <venv>/bin/Activate.psl

# windows
> - cmd.exe: <venv>\Scripts\activate.bat
> - PowerShell: <venv>\Scripts\Activate.psl

## challenge: 1
> - 프로그래밍 언어에 대한 시를 쓰는 데 특화된 체인과 시를 설명하는 데 특화된 체인을 만드세요.
> - LCEL을 사용해 두 체인을 서로 연결합니다.
> - 최종 체인은 프로그래밍 언어의 이름을 받고 시와 그 설명으로 응답해야 합니다.

## challenge: 2
> - 영화 이름을 가지고 감독, 주요 출연진, 예산, 흥행 수익, 영화의 장르, 간단한 시놉시스 등 영화에 대한 정보로 답장하는 체인을 만드세요.
> - LLM은 항상 동일한 형식을 사용하여 응답해야 하며, 이를 위해서는 원하는 출력의 예시를 LLM에 제공해야 합니다.

## challenge: 3
> - 메모리 클래스 중 하나를 사용하는 메모리로 LCEL 체인을 구현
> - "탑건" -> "🛩️👨‍✈️🔥". "대부" -> "👨‍👨‍👦🔫🍝"

## challenge: 4
> - Stuff Documents 체인을 사용하여 완전한 RAG 파이프라인을 구현
> - 체인에 ConversationBufferMemory 부여
> - 참고 문서: https://gist.github.com/serranoarevalo/5acf755c2b8d83f1707ef266b82ea223
> - 질문: Aaronson 은 유죄인가요? | 그가 테이블에 어떤 메시지를 썼나요? | Julia 는 누구인가요?
