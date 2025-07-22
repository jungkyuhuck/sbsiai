import pickle

def main():
    try:
        with open("sbs_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        print(f"✅ 총 메타데이터 개수: {len(metadata)}\n")

        # 🔎 전체 항목 모두 출력 (최대 50개만 출력하도록 제한)
        for i, item in enumerate(metadata[:3000], 1):
            print(f"{i}. {item}")

        # 참고: 너무 많으면 슬라이싱 범위를 조절하거나 저장하는 방식으로 변경
        # 예) metadata[:100], metadata[-10:], etc.

    except FileNotFoundError:
        print("❌ 'sbs_metadata.pkl' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
