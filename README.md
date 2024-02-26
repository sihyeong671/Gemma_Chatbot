# Gemma Chatbot with PyTorch and Streamlit

you can download model below link<br/>
<https://www.kaggle.com/models/google/gemma/frameworks/pyTorch>

Gemma PyTorch library<br/>
<https://github.com/google/gemma_pytorch>

Gemma PyTorch tutorial<br/>
<https://ai.google.dev/gemma/docs/pytorch_gemma>

## Get Start

1. download google/gemma/2b-it from [Model Variations/PyTorch] section of <https://www.kaggle.com/models/google/gemma/frameworks/pyTorch/variations/2b-it>

2. 다운로드한 압축파일을 압축해제한 뒤, 작업 디렉토리로 옮긴다.
3. 작업 디렉토리 내에서 `git clone https://github.com/google/gemma_pytorch.git`
4. (선택사항) 가상환경 설치
   4-1. `python3 -m venv .venv`
   4-2. `.venv\Scripts\activate` in Windows
   4-3. `source .venv/bin/activate` in macOS/Linux
5. `pip install -r requirements.txt`

---

## How To use

```sh
streamlit run main.py
```

![img](./img/chatbot.png)

---

## Environment

- MACHINE_TYPE == 'cpu': 매우 느림
- MACHINE_TYPE == 'cuda': 그래픽카드 성능에 따라 다르지만 'cpu'에 비하면 훨씬 빠름
