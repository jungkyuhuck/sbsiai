# app.py
from fastapi import FastAPI
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import numpy as np
import faiss, pickle
import os
from dotenv import load_dotenv
load_dotenv()

# 환경변수에서 API 키 읽기
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 생성
client = OpenAI(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ✅ FAISS‧메타‧벡터 로드
faiss_index = faiss.read_index("sbs_index.faiss")
with open("sbs_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
all_vectors = np.load("sbs_vectors.npy")  # ▲▲ 추가: 벡터들

# ✅ 등장인물 리스트
KNOWN_CHARACTERS = [
    "김도기", "강하나", "장성철", "고은", "최경구", "백성미",
    "박양진", "조도철", "우재연", "서현수", "온하준"
]

# ✅ 등장인물 필터 함수
def extract_known_character(question: str) -> Optional[str]:
    for name in KNOWN_CHARACTERS:
        if name in question:
            return name
    return None

# ✅ 임베딩
def get_embedding(text: str) -> list:
    res = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

# ✅ FAISS 검색 + 유사도 필터링
SIM_THRESHOLD = 0.75  # 코사인 유사도 임계값

def query_faiss_best_match(question: str, top_k: int = 3) -> str:
    q_emb = get_embedding(question)
    q_emb = np.array(q_emb, dtype="float32")
    q_emb /= np.linalg.norm(q_emb)

    D, I = faiss_index.search(q_emb[np.newaxis, :], top_k)
    best_idx = I[0][0]
    best_vec = all_vectors[best_idx]
    sim = float(np.dot(q_emb, best_vec))

    if sim < SIM_THRESHOLD:
        return ""
    return "\n".join(metadata[i] for i in I[0] if i < len(metadata))

# ✅ GPT 호출
def ask_chatbot(user_q: str, ctx: str) -> str:
    sys_prompt = (
        "너는 SBS 드라마 '모범택시' 전용 캐릭터 AI 챗봇이야. "
        "시즌1~2에 등장한 주요 인물들의 성격, 배경, 명대사, 사건들을 학습한 너는 질문자의 요청에 따라 "
        "드라마의 맥락에 맞게 **친절하고 몰입감 있게 정보를 제공**해야 해.\n\n"
        "🧠 중요한 원칙: 아래 참고 문장은 방송 메타데이터에서 추출한 고신뢰 정보야. "
        "답변은 반드시 이 내용을 중심으로 하고, 너의 상식이나 외부 지식은 참고 수준으로만 활용해야 해.\n\n"
        "❌ 등장인물이나 사건이 데이터에 없으면 절대 상상해서 말하지 마.\n"
    "❌ 질문에 나온 인물이 KNOWN_CHARACTERS 목록에 없다면 절대 답하지 마. 모른다고 답해.\n"
    "❗ 등장인물인지 불확실하면 '드라마에 등장하지 않는다'고 말해야 해.\n"
        "- 질문자가 '너는 어떤 챗봇이야?'라고 물으면: "
        "'DX팀1 규혁, 희선, 화정, 주리가 개발한 SBS 방송 메타데이터 기반 AI 챗봇이야. 특히 드라마 <모범택시>를 중심으로 학습했어.'\n\n"
        "등장인물 관련 질문이면 배경, 성격, 대사 중심으로. 줄거리나 사건이면 회차별 맥락 위주로. "
        "모호하면 어떤 인물/회차/사건인지 먼저 확인해줘. "
        "말투는 팬에게 말하듯 몰입감 있게. 김도기라면 담백하고 묵직하게."
    )

    if ctx:
        sys_prompt += f"\n📚 반드시 참고할 데이터:\n{ctx}\n"

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_q}
        ]
    )
    return resp.choices[0].message.content.strip()

# ✅ API 엔드포인트
@app.get("/youtube/chat")
async def chat(question: Optional[str] = ""):
    if not question:
        return {"error": "❗ /youtube/chat?question=... 형식으로 호출하세요."}

    # ✅ 등장인물 유효성 검사
    char_name = extract_known_character(question)
    if ("누구" in question or "어떤 인물" in question) and not char_name:
        return {
            "question": question,
            "context_used": "",
            "answer": "죄송하지만 질문에 나온 인물은 드라마에 등장하지 않습니다."
        }

    ctx = query_faiss_best_match(question)
    if not ctx:
        return {
            "question": question,
            "context_used": "",
            "answer": "데이터에 기반한 정보가 없어 정확히 말씀드리기 어렵습니다."
        }

    answer = ask_chatbot(question, ctx)
    return {"question": question, "context_used": ctx, "answer": answer}

# ✅ 기본 루트 엔드포인트
@app.get("/youtube")
async def root():
    return {"message": "Hello World"}

@app.get("/")
async def root2():
    return {"message": "Hello World2"}
