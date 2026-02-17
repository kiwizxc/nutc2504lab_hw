import os
import glob
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. 必要套件匯入
# ==========================================
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==========================================
# 2. 設定 API 與 模型資訊
# ==========================================

# 初始化 LLM (改寫 Query 和生成答案用)
llm_client = OpenAI(
    base_url="https://ws-03.wade0426.me/v1", 
    api_key="sk-dummy-key"
)
LLM_MODEL = "/models/gpt-oss-120b"

# 初始化 Embedding 模型 (將文字轉向量用)
# all-MiniLM-L6-v2 的維度是 384
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_SIZE = 384 

# 初始化 Qdrant 客戶端
# 確保你的 Docker 已經執行: sudo docker run -p 6333:6333 ...
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "workshop_day6_cw03"


# ==========================================
# 3. 資料庫建立與寫入 (Ingestion)
# ==========================================

def init_qdrant_collection():
    """重新建立 Qdrant 集合"""
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )
    print(f"已建立 Qdrant 集合: {collection_name}")

def load_and_upload_data():
    """讀取 data_*.txt 並寫入資料庫"""
    # 檢查是否已有 data_*.txt
    files = glob.glob("data_*.txt")
    if not files:
        print("警告: 找不到 data_*.txt 檔案，跳過資料寫入步驟。")
        return

    print(f"發現 {len(files)} 個資料檔，準備寫入資料庫...")
    
    # 初始化集合
    init_qdrant_collection()
    
    documents = []
    
    # 讀取檔案與切塊
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            # 簡單切塊 (每 300 字一塊)
            chunk_size = 300
            overlap = 50
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk) > 50:
                    documents.append(chunk)

    # 轉向量
    print("正在計算向量 (Embeddings)...")
    embeddings = embedding_model.encode(documents).tolist()
    
    # 上傳 Qdrant
    points = []
    for idx, (doc, vector) in enumerate(zip(documents, embeddings)):
        points.append(PointStruct(
            id=idx,
            vector=vector,
            payload={"text": doc} # 將原始文字存入 payload
        ))
        
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"成功寫入 {len(points)} 筆資料！")


# ==========================================
# 4. 核心功能函式
# ==========================================

def query_rewrite(original_query):
    """[CW-03 重點] 使用 LLM 改寫問題"""
    system_prompt = """
    你是一個搜尋優化助理。請將使用者的查詢修正錯字並轉化為精確的搜尋關鍵字。
    直接輸出優化後的查詢字串即可，不要解釋。
    """
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_query}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"改寫失敗: {e}")
        return original_query

def rag_search(rewritten_query):
    """
    [CW-03 重點] 使用 query_points 進行搜尋 (修正版)
    """
    # 1. 將改寫後的問題轉向量
    vector = embedding_model.encode([rewritten_query])[0].tolist()
    
    # 2. 搜尋 (使用 query_points 替代 search)
    search_response = qdrant_client.query_points(
        collection_name=collection_name,
        query=vector,  # 參數名稱是 query
        limit=3        # 取 Top 3
    )
    
    # 3. 取出結果 (注意要加上 .points)
    results = search_response.points
    
    # 4. 提取 payload 中的文字
    contexts = [hit.payload['text'] for hit in results]
    return "\n\n".join(contexts)

def generate_answer(original_query, context):
    """根據搜尋結果回答問題"""
    system_prompt = f"""
    你是 AI 助理。請根據以下 [參考資料] 回答使用者的問題。
    如果資料不足，請回答「無法從資料中找到答案」。
    
    [參考資料]
    {context}
    """
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_query}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成失敗: {e}"

# ==========================================
# 5. 主程式
# ==========================================

def main():
    # 1. 嘗試寫入資料 (如果資料庫是空的)
    # 你可以註解掉這一行如果你不想每次都重新寫入資料
    load_and_upload_data()
    
    input_csv = "Re_Write_questions.csv"
    output_csv = "Re_Write_answers.csv"
    
    # 建立範例 CSV (防呆)
    if not os.path.exists(input_csv):
        print(f"建立測試檔案 {input_csv}...")
        df_demo = pd.DataFrame({"question": ["請假流承", "我想換部門"]})
        df_demo.to_csv(input_csv, index=False)

    # 讀取 CSV
    df = pd.read_csv(input_csv)
    print(f"開始處理 {len(df)} 個問題...")
    
    # 自動偵測問題欄位 (處理 Key Error)
    question_col = 'question'
    if 'question' not in df.columns:
        question_col = df.columns[0] # 若沒有 question 欄位，就拿第一欄
        print(f"提示: 使用 '{question_col}' 作為問題欄位")

    rewritten_queries = []
    answers = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        q = str(row[question_col])
        
        # Step A: 改寫
        rw_q = query_rewrite(q)
        
        # Step B: 搜尋 (使用修正後的 rag_search)
        context = rag_search(rw_q)
        
        # Step C: 回答
        ans = generate_answer(q, context)
        
        rewritten_queries.append(rw_q)
        answers.append(ans)
        
    # 存檔
    df['rewritten_query'] = rewritten_queries
    df['answer'] = answers
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"完成！結果已儲存至 {output_csv}")

if __name__ == "__main__":
    main()