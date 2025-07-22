import openai
import numpy as np
import faiss
import pickle
import kss
from pymongo import MongoClient
import time
import os
from dotenv import load_dotenv  # ✅ 추가

# ✅ .env 파일에서 OpenAI API 키 불러오기
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ 임베딩 함수
def get_embedding(text: str) -> list:
    try:
        res = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return res["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ 임베딩 실패: {text[:30]}... → {e}")
        return None

# ✅ MongoDB → 텍스트 추출
def fetch_episodes_and_characters(uri: str, db: str, coll: str) -> list:
    client = MongoClient(uri)
    col = client[db][coll]
    texts = []

    for doc in col.find({}):
        title = doc.get("title") or doc.get("program_title") or "UNKNOWN"
        season = doc.get("season", 0)

        # episodes
        for ep in doc.get("episodes", []):
            ep_no = ep.get("episode", "UNKNOWN")
            # summary
            summary = ep.get("summary", "")
            for s in kss.split_sentences(summary):
                texts.append(f"[{title} S{season}E{ep_no}] 줄거리: {s.strip()}")
            # vtt
            vtt = ep.get("vtt", "")
            for s in kss.split_sentences(vtt):
                texts.append(f"[{title} S{season}E{ep_no}] 대사: {s.strip()}")

        # characters
        for ch in doc.get("characters", []):
            name = ch.get("name", "UNKNOWN")
            for s in kss.split_sentences(ch.get("description", "")):
                texts.append(f"[{title} 캐릭터 {name}] 설명: {s.strip()}")

    return texts

# ✅ 인덱스 생성 및 저장
def build_and_save_index(texts: list,
                         faiss_path: str = "sbs_index.faiss",
                         meta_path: str = "sbs_metadata.pkl",
                         vec_path: str = "sbs_vectors.npy"):
    vectors, metadata = [], []
    for t in texts:
        emb = get_embedding(t)
        if emb:
            vectors.append(emb)
            metadata.append(t)
        time.sleep(0.05)  # API 부하 완화

    if not vectors:
        print("❌ 임베딩된 문장이 없습니다.")
        return

    vecs = np.array(vectors).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # 🔄 코사인 유사도 정규화

    index = faiss.IndexFlatIP(vecs.shape[1])  # 코사인 유사도용 내적 인덱스
    index.add(vecs)

    # ▶ 저장
    faiss.write_index(index, faiss_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    np.save(vec_path, vecs)

    print(f"✅ 벡터 {len(vecs)}개, 메타데이터 {len(metadata)}개 저장 완료")

# ===== 실행 =====
if __name__ == "__main__":
    URI = "mongodb://43.201.154.108:27017"
    DB = "aiproject"
    COLL = "deep learning_data"
    txts = fetch_episodes_and_characters(URI, DB, COLL)
    build_and_save_index(txts)
