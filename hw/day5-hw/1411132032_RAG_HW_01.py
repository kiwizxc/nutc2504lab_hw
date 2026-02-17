import os
import csv
import glob
import requests
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# ================= 設定區 =================
STUDENT_ID = "1411132032"  # 【請記得修改學號】
DATA_FOLDER = "day5"            # 資料夾名稱
OUTPUT_CSV = f"{STUDENT_ID}_RAG_HW_01.csv"

# API 設定
EMBEDDING_API_URL = "https://ws-04.wade0426.me/embed"
SCORE_API_URL = "https://hw-01.wade0426.me/submit_answer"

# Qdrant 設定 (修改為 Docker 模式)
# 請確保 Docker 容器已在背景執行 (sudo docker run ...)
client = QdrantClient(url="http://localhost:6333") 

COLLECTION_NAME = "day5_homework"
VECTOR_SIZE = 4096

# ================= 函數定義 (保持不變) =================

def get_embedding(text: str) -> List[float]:
    """呼叫 API 獲取向量"""
    try:
        payload = {"texts": [text], "normalize": True, "batch_size": 1}
        response = requests.post(EMBEDDING_API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()['embeddings'][0]
        else:
            print(f"Embedding Error: {response.text}")
            return [0.0] * VECTOR_SIZE
    except Exception as e:
        print(f"Embedding Exception: {e}")
        return [0.0] * VECTOR_SIZE

def get_score(q_id: int, answer: str) -> float:
    """呼叫 API 獲取分數"""
    try:
        payload = {"q_id": q_id, "student_answer": answer}
        response = requests.post(SCORE_API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get('score', 0.0)
        else:
            print(f"Score API Error (Q{q_id}): {response.text}")
            return 0.0
    except Exception as e:
        print(f"Score API Exception: {e}")
        return 0.0

def load_data(folder_path: str) -> List[Dict]:
    documents = []
    file_list = glob.glob(os.path.join(folder_path, "data_*.txt"))
    print(f"找到 {len(file_list)} 個資料檔案")
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.append({"source": file_name, "text": f.read()})
    return documents

# --- 切塊策略 (保持不變) ---
def chunk_fixed_size(text: str, size: int = 300) -> List[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]

def chunk_sliding_window(text: str, size: int = 300, overlap: int = 60) -> List[str]:
    step = size - overlap
    if step <= 0: step = 1
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i+size]
        if len(chunk) > size * 0.5:
            chunks.append(chunk)
    return chunks

def chunk_semantic(text: str) -> List[str]:
    import re
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(para) > 500:
            sentences = re.split(r'(?<=[。？！])', para)
            for sent in sentences:
                if len(current_chunk) + len(sent) < 500:
                    current_chunk += sent
                else:
                    if current_chunk: chunks.append(current_chunk)
                    current_chunk = sent
        else:
            if len(current_chunk) + len(para) < 500:
                current_chunk += (para + "\n")
            else:
                if current_chunk: chunks.append(current_chunk)
                current_chunk = para + "\n"
    if current_chunk: chunks.append(current_chunk)
    return chunks

# ================= 主程式 =================
def main():
    # 1. 初始化 Qdrant Collection
    # 注意：這會刪除舊的 collection 重新建立，適合這份作業的性質
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    
    # 2. 讀取與切塊
    docs = load_data(DATA_FOLDER)
    if not docs:
        print(f"錯誤：找不到檔案，請確認 {DATA_FOLDER} 資料夾是否存在且有 .txt 檔")
        return

    methods = [
        ("固定大小", chunk_fixed_size),
        ("滑動視窗", chunk_sliding_window),
        ("語意切塊", chunk_semantic)
    ]
    
    print("\n[1/3] 建立向量索引中...")
    points = []
    p_id = 0
    for method_name, split_func in methods:
        print(f"  - 處理方法: {method_name}")
        for doc in docs:
            chunks = split_func(doc['text'])
            for chunk in chunks:
                if not chunk.strip(): continue
                point = PointStruct(
                    id=p_id,
                    vector=get_embedding(chunk),
                    payload={"method": method_name, "source": doc['source'], "text": chunk}
                )
                points.append(point)
                p_id += 1
                # 批次寫入
                if len(points) >= 50:
                    client.upsert(COLLECTION_NAME, points)
                    points = []
    if points: client.upsert(COLLECTION_NAME, points)

    # 3. 讀取問題
    q_file = os.path.join(DATA_FOLDER, "questions.csv")
    if not os.path.exists(q_file): q_file = "questions.csv"
    
    questions = []
    # 檢查檔案是否存在
    if not os.path.exists(q_file):
        print(f"錯誤：找不到 {q_file}")
        return

    with open(q_file, 'r', encoding='utf-8-sig') as f:
        questions = list(csv.DictReader(f))

    # 4. 回答與評分
    print("\n[2/3] 開始回答問題並評分...")
    results = []
    row_id = 1
    
    score_sums = {m[0]: 0.0 for m in methods}
    score_counts = {m[0]: 0 for m in methods}

    for q in questions:
        q_id = int(q['q_id'])
        q_vec = get_embedding(q['questions'])
        print(f"  - Q{q_id}...", end="\r")
        
        for method_name, _ in methods:
            # 搜尋
            search_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=q_vec,
                query_filter=Filter(must=[FieldCondition(key="method", match=MatchValue(value=method_name))]),
                limit=1
            ).points
            
            if search_result:
                best = search_result[0]
                score = get_score(q_id, best.payload['text'])
                
                results.append({
                    "id": row_id, "q_id": q_id, "method": method_name,
                    "retrieve_text": best.payload['text'], "score": score, "source": best.payload['source']
                })
                
                score_sums[method_name] += score
                score_counts[method_name] += 1
                row_id += 1

    # 5. 輸出 CSV
    keys = ["id", "q_id", "method", "retrieve_text", "score", "source"]
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[3/3] 完成！結果已存至 {OUTPUT_CSV}")

    # 6. 印出平均分數
    print("\n" + "="*30)
    print("      各方法平均分數統計      ")
    print("="*30)
    for method_name, _ in methods:
        count = score_counts[method_name]
        avg = score_sums[method_name] / count if count > 0 else 0.0
        print(f"{method_name}: {avg:.4f} 分")
    print("="*30)

if __name__ == "__main__":
    main()