# train_embeddings.py
import requests
import re
import openai
import numpy as np
import faiss
import pickle
import kss  # 한국어 문장 분리
import os
from dotenv import load_dotenv

# ✅ .env 파일에서 API 키 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # .env에 저장된 키 사용

def get_embedding(text: str) -> list:
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ 임베딩 실패: {text[:30]}... → {e}")
        return None

def build_faiss_from_episode_list(episodes: list):
    all_vectors = []
    all_metadata = []
    seen_texts = set()

    for ep in episodes:
        title = ep.get("program_title", "UNKNOWN")
        episode = ep.get("episode_no", "UNKNOWN")
        summary = ep.get("summary", "")

        print(f"🎬 {title} {episode} 처리 중...")

        if summary.strip():
            sentences = kss.split_sentences(summary.strip())
            for s in sentences:
                text = f"[{title} {episode}] 줄거리: {s.strip()}"
                if text in seen_texts:
                    continue
                seen_texts.add(text)

                emb = get_embedding(text)
                if emb:
                    all_metadata.append(text)
                    all_vectors.append(emb)

    if not all_vectors:
        print("❌ 저장할 벡터 없음.")
        return

    vectors_np = np.array(all_vectors).astype("float32")
    dim = len(vectors_np[0])
    index = faiss.IndexFlatL2(dim)
    index.add(vectors_np)

    faiss.write_index(index, "sbs_index.faiss")
    with open("sbs_metadata.pkl", "wb") as f:
        pickle.dump(all_metadata, f)

    print(f"✅ 저장 완료: {len(all_metadata)} 문장 → FAISS + 메타데이터")

# ✨ 훈련용 에피소드 정의
episodes = [
    {
        "program_title": "모범택시",
        "episode_no": 0,
        "summary": """
김도기 (이제훈 역)

前 육사, 특수부대(육군특수전사령부 707특수임무단) 장교.
現 무지개 운수의 택시기사.
...
과연 도기는 그 깊은 터널을 빠져나올 수 있을까?
"""
    },
    {
        "program_title": "모범택시",
        "episode_no": 1,
        "summary": """
오늘 우리는 회의를 했습니다. 주제는 AI 스터디입니다.
SBSI AI 스터디 멤버는 희선, 화정, 규리, 규혁, 주리입니다.
...
김도기한테 고민상담을 하고 싶으면 말해. 뭐가 필요해?
"""
    }
]

if __name__ == "__main__":
    build_faiss_from_episode_list(episodes)
