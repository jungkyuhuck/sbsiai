import openai
import numpy as np
import faiss
import pickle
import kss
from pymongo import MongoClient
import os
import time

# ✅ OpenAI API 키 설정
openai.api_key = ""

# ✅ 임베딩 생성 함수
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

# ✅ MongoDB에서 데이터 추출 및 텍스트 분해
def fetch_episodes_and_characters(mongo_uri: str, db_name: str, collection_name: str) -> list:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    cursor = collection.find({})
    all_texts = []

    for doc in cursor:
        program_title = doc.get("title") or doc.get("program_title") or "UNKNOWN"
        season = doc.get("season", 0)

        # 📺 episodes 처리
        episodes = doc.get("episodes", [])
        for ep in episodes:
            ep_no = ep.get("episode", "UNKNOWN")

            # 1) summary
            summary = ep.get("summary", "").strip()
            if summary:
                sentences = kss.split_sentences(summary)
                for sent in sentences:
                    all_texts.append(f"[{program_title} S{season}E{ep_no}] 줄거리: {sent.strip()}")

            # 2) vtt
            vtt = ep.get("vtt", "").strip()
            if vtt:
                vtt_sentences = kss.split_sentences(vtt)
                print(f"\n🎬 [DEBUG] {program_title} S{season}E{ep_no} - 대사 문장 수: {len(vtt_sentences)}")
                for i, sent in enumerate(vtt_sentences):
                    preview = sent.strip()[:60].replace("\n", " ")
                    print(f"  🔹 ({i + 1}) {preview}...")
                    all_texts.append(f"[{program_title} S{season}E{ep_no}] 대사: {sent.strip()}")

        # 👤 characters 처리
        characters = doc.get("characters", [])
        for ch in characters:
            name = ch.get("name", "UNKNOWN")
            description = ch.get("description", "").strip()
            if description:
                sentences = kss.split_sentences(description)
                for sent in sentences:
                    all_texts.append(f"[{program_title} 캐릭터 {name}] 설명: {sent.strip()}")

    return all_texts

# ✅ FAISS 인덱스 생성 및 저장
def build_and_save_index(texts: list,
                         faiss_path: str = "C:/Users/apf_temp_admin/PycharmProjects/newai/sbs_index.faiss",
                         meta_path: str = "C:/Users/apf_temp_admin/PycharmProjects/newai/sbs_metadata.pkl"):
    vectors = []
    metadata = []
    seen = set()
    count_total = 0
    count_embedded = 0
    count_dialogue = 0

    # 디렉토리 없으면 생성
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)

    for text in texts:
        count_total += 1

        # 중복 제거
        if text in seen:
            continue
        seen.add(text)

        # 대사 여부 확인
        if "대사:" in text:
            count_dialogue += 1

        # OpenAI 임베딩 시도
        embedding = get_embedding(text)
        if embedding:
            vectors.append(embedding)
            metadata.append(text)
            count_embedded += 1
        else:
            print(f"⚠️ 임베딩 실패: {text[:40]}")

        time.sleep(0.1)

    # 벡터 저장
    if not vectors:
        print("❌ 저장할 벡터 없음.")
        return

    np_vectors = np.array(vectors).astype("float32")
    dim = len(np_vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np_vectors)

    faiss.write_index(index, faiss_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print("\n✅ 최종 저장 완료")
    print(f"총 처리 텍스트 수: {count_total}")
    print(f"임베딩 성공 수: {count_embedded}")
    print(f"대사 문장 포함 수: {count_dialogue}")
    print(f"최종 저장된 메타데이터 수: {len(metadata)}")

# ✅ 실행
if __name__ == "__main__":
    MONGO_URI = "mongodb://43.201.154.108:27017"
    DB_NAME = "aiproject"
    COLLECTION_NAME = "deep learning_data"

    texts = fetch_episodes_and_characters(MONGO_URI, DB_NAME, COLLECTION_NAME)
    build_and_save_index(texts)
