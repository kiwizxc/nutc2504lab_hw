import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# 1. 連接 Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "cw01_collection"

# ⭐ 重要修正：如果已存在但維度不對，直接刪除重來，確保維度是 4096
if client.collection_exists(collection_name):
    print(f"检测到舊的 {collection_name}，正在重置以更正維度...")
    client.delete_collection(collection_name=collection_name)

# 重新建立正確維度 (4096) 的 Collection
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)
print(f"✅ Collection '{collection_name}' (維度: 4096) 建立完成")

# 2. & 3. 獲取向量 (這部分你的代碼很正確)
def get_embeddings(texts):
    API_URL = "https://ws-04.wade0426.me/embed"
    payload = {
        "texts": texts,
        "task_description": "檢索技術文件",
        "normalize": True
    }
    response = requests.post(API_URL, json=payload)
    return response.json()["embeddings"]

data_contents = [
    "RAG 技術能有效結合外部知識庫與大型語言模型。",
    "向量資料庫 Qdrant 支援高效的相似度檢索。",
    "Embedding 是將文字轉化為高維度空間向量的過程。",
    "分塊策略 (Chunking) 影響了檢索結果的準確性與上下文完整性。",
    "餘弦相似度 (Cosine Similarity) 是常用的向量距離計算方法。",
    "GitHub 是程式碼版本管理與協作開發的首選平台。"
]

print("🚀 正在呼叫 API 轉換向量...")
vectors = get_embeddings(data_contents)

# 4. 寫入 Points (Upsert)
print("📥 正在將 Points 寫入 Qdrant...")
points = [
    PointStruct(
        id=i, 
        vector=vectors[i], 
        payload={"text": data_contents[i], "source": "CW01_Task"}
    ) for i in range(len(data_contents))
]

client.upsert(collection_name=collection_name, points=points)
print("✅ 資料寫入成功！")

# 5. 召回內容 (Recall)
def query_task(query_text):
    query_vector = get_embeddings([query_text])[0]
    # 注意：新版 Qdrant 建議使用 query_points
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3
    ).points
    
    print(f"\n🔍 搜尋問題: '{query_text}'")
    for hit in results:
        print(f"- 內容: {hit.payload['text']} (分數: {hit.score:.4f})")

# 測試召回
query_task("如何解決 LLM 幻覺問題？")