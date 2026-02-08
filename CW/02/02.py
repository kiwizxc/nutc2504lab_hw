import os
import csv
import requests
import re
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# --- 1. 環境配置 ---
client = QdrantClient(host="localhost", port=6333)
collection_name = "cw02_final_collection"
API_URL = "https://ws-04.wade0426.me/embed"

# 建立假資料：如果沒有 text.txt，建立一個
if not os.path.exists("text.txt"):
    with open("text.txt", "w", encoding="utf-8") as f:
        f.write("這是一個測試文本。它用來測試切塊功能。\n\n這是第二段落，用來測試語意分割的效果。")

# 建立假資料：如果沒有 table 資料夾或裡面沒檔案，建立一個 csv 測試
if not os.path.exists("table"):
    os.makedirs("table")
    
if not os.listdir("table"):
    csv_content = [
        ["產品", "價格", "庫存", "備註"],
        ["蘋果", "30", "100", "新鮮到貨"],
        ["香蕉", "15", "50", "來自旗山"],
        ["橘子", "25", "80", "季節限定"]
    ]
    with open("table/sample_table.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_content)
    print("⚠️ 偵測到 table 資料夾為空，已自動建立 sample_table.csv 供測試用。")

# 讀取主要文本
try:
    with open("text.txt", "r", encoding="utf-8") as f:
        source_text = f.read()
except FileNotFoundError:
    source_text = ""

# --- 2. 實作：固定切塊 (Fixed-size Chunking) ---
def fixed_size_chunking(text, chunk_size=200):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# --- 3. 實作：語意滑動視窗 (符合圖片下半部要求) ---
def semantic_sliding_window(text, chunk_size=250, overlap=50):
    chunks = []
    sentences = re.split(r'(。|！|？|\n+)', text)
    
    current_chunk = ""
    combined_sentences = []
    temp_sent = ""
    for s in sentences:
        temp_sent += s
        if re.search(r'(。|！|？|\n+)', s) or len(s.strip()) == 0:
            if temp_sent.strip():
                combined_sentences.append(temp_sent)
            temp_sent = ""
    if temp_sent: combined_sentences.append(temp_sent)

    for sentence in combined_sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            chunks.append(current_chunk)
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + sentence
            
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# --- 4. [強化版] 實作：處理 table 資料夾 ---
def csv_to_markdown_table(file_path):
    """將 CSV 轉換為 Markdown 表格字串，讓 LLM 更容易理解"""
    try:
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows: return ""
            
            # 製作 Markdown Header
            header = "| " + " | ".join(rows[0]) + " |"
            separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
            
            # 製作內容
            body = []
            for row in rows[1:]:
                body.append("| " + " | ".join(row) + " |")
                
            return f"{header}\n{separator}\n" + "\n".join(body)
    except Exception as e:
        print(f"CSV 解析失敗: {e}")
        return ""

def process_table_folder(folder_path="table"):
    table_chunks = []
    if not os.path.exists(folder_path):
        print(f"⚠️ 找不到 {folder_path} 資料夾，跳過表格處理")
        return []
    
    print(f"📂 正在處理 {folder_path} 資料夾...")
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        content = ""
        
        # 忽略系統檔案
        if filename.startswith("."): continue
        
        try:
            # 針對不同副檔名做處理
            if filename.lower().endswith(".csv"):
                # CSV 轉 Markdown
                raw_csv = csv_to_markdown_table(file_path)
                if raw_csv:
                    content = raw_csv
                    print(f"  - 已轉換 CSV: {filename}")
            
            elif filename.lower().endswith((".html", ".md", ".txt")):
                # 純文字類直接讀取
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    print(f"  - 已讀取文字檔: {filename}")
            
            else:
                print(f"  - 跳過不支援的檔案格式: {filename}")
                continue

            if content:
                # 加上來源標示，這對 RAG 很重要
                formatted_content = f"【表格來源: {filename}】\n{content}"
                table_chunks.append(formatted_content)
                
        except Exception as e:
            print(f"讀取 {filename} 出錯: {e}")
            
    return table_chunks

# --- 5. 向量化函式 ---
def get_embeddings(texts):
    if not texts: return []
    batch_size = 5
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        payload = {
            "texts": batch,
            "task_description": "檔案分塊與表格檢索實作",
            "normalize": True
        }
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                all_embeddings.extend(response.json()["embeddings"])
            else:
                print(f"Embedding API Error: {response.text}")
                # 失敗時回傳空向量 (除錯用)
                all_embeddings.extend([[0.0]*4096] * len(batch))
        except Exception as e:
            print(f"API Connection Error: {e}")
            all_embeddings.extend([[0.0]*4096] * len(batch))
            
    return all_embeddings

# --- 6. 輔助函式：執行評估 ---
def evaluate_method(method_name, chunks, query_text):
    if not chunks:
        print(f"[{method_name}] 沒有區塊可處理，跳過。")
        return

    print(f"\n🧪 正在評估方法: {method_name} (共 {len(chunks)} 個區塊)")
    
    vectors = get_embeddings(chunks)
    
    temp_col = f"temp_{method_name}"
    if client.collection_exists(temp_col):
        client.delete_collection(temp_col)
    
    client.create_collection(
        collection_name=temp_col,
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
    )
    
    points = [
        PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=temp_col, points=points)
    
    query_vec = get_embeddings([query_text])[0]
    results = client.query_points(collection_name=temp_col, query=query_vec, limit=1).points
    
    if results:
        preview_text = results[0].payload['text'][:50].replace('\n', ' ')
        print(f"   👉 檢索結果: {preview_text}...")
        print(f"   👉 分數 (Score): {results[0].score:.4f}")
    else:
        print("   👉 無搜尋結果")

# ================= 執行流程 =================

if __name__ == "__main__":
    # 1. 準備切塊資料
    print("--- 1. 執行切塊 ---")
    fixed_chunks = fixed_size_chunking(source_text)
    semantic_chunks = semantic_sliding_window(source_text)
    
    # 這裡會使用強化版的表格處理函式
    table_chunks = process_table_folder("table")

    # 2. 比較兩種切塊方法
    print("\n--- 2. 比較切塊方法 ---")
    test_query = "請說明本文的核心重點是什麼？"
    
    evaluate_method("固定切塊(Fixed)", fixed_chunks, test_query)
    evaluate_method("滑動視窗切塊(sliding+Semantic)", semantic_chunks, test_query)

    # 3. 建立最終作業資料庫
    print("\n--- 3. 建立最終資料庫 (CW/02) ---")
    final_chunks = semantic_chunks + table_chunks

    if final_chunks:
        print(f"🚀 正在寫入 {len(final_chunks)} 筆資料到 {collection_name}...")
        final_vectors = get_embeddings(final_chunks)

        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        )

        final_points = [
            PointStruct(id=i, vector=final_vectors[i], payload={"text": final_chunks[i]})
            for i in range(len(final_chunks))
        ]

        client.upsert(collection_name=collection_name, points=final_points)
        print(f"✅ 資料已存入 {collection_name}")

        # 4. [新增] 針對表格的專屬測試
        print("\n--- 4. 表格檢索測試 ---")
        # 如果你剛剛自動生成了香蕉的資料，這裡應該要能搜到
        table_query = "香蕉的價格與庫存是多少？" 
        print(f"測試問題: {table_query}")
        
        t_vec = get_embeddings([table_query])[0]
        res = client.query_points(collection_name=collection_name, query=t_vec, limit=1).points
        if res:
            table_preview = res[0].payload['text'][:100].replace('\n', ' ')
            print(f"👉 搜尋結果: {table_preview}...")
            print(f"👉 分數: {res[0].score:.4f}")
        else:
            print("❌ 找不到表格相關資料")
    else:
        print("⚠️ 沒有任何資料塊被產生，請檢查 source_text 或 table 資料夾")