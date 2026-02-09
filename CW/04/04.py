import os
import glob
import torch
import pandas as pd
from tqdm import tqdm

# Qdrant & Embedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# LLM
from openai import OpenAI

# Reranker 專用
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. 設定區 (Configuration)
# ==========================================

# Qdrant 設定
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "workshop_day6_cw04"

# Embedding 模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_SIZE = 384

# LLM 設定
llm_client = OpenAI(base_url="https://ws-03.wade0426.me/v1", api_key="sk-dummy-key")
LLM_MODEL = "/models/gpt-oss-120b"

# ==========================================
# 2. 載入 Qwen3-Reranker 模型
# ==========================================

# [已修正] 指向你提供的路徑 (WSL 格式)
model_path = "/home/kiwi/test_folder/day6/Qwen3-Reranker-0.6B"

print(f"正在載入 Reranker 模型: {model_path} ...")

# 檢查路徑是否存在
if not os.path.exists(model_path):
    print(f"錯誤：找不到模型路徑 {model_path}")
    print("請確認資料夾名稱是否正確？")
    exit()

try:
    reranker_tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    reranker_model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).eval()
    
    if torch.cuda.is_available():
        reranker_model = reranker_model.to("cuda")
        print("Reranker 已載入至 GPU")
    else:
        print("警告: 未偵測到 GPU，Reranker 將使用 CPU 執行 (速度較慢)")

except Exception as e:
    print(f"Reranker 模型載入失敗: {e}")
    exit()

# Reranker 參數
token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
max_reranker_length = 4096

# Prompt 模板
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n"
prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)


# ==========================================
# 3. Reranker 核心函式
# ==========================================

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = '根據查詢檢索相關文件'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

@torch.no_grad()
def get_rerank_scores(query, candidate_docs):
    """計算 Query 與每個 Document 的相關性分數"""
    if not candidate_docs:
        return []

    input_texts = [format_instruction(None, query, doc) for doc in candidate_docs]
    
    # 手動 Batch Tokenization
    batch_input_ids = []
    for text in input_texts:
        text_ids = reranker_tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_reranker_length - len(prefix_tokens) - len(suffix_tokens))
        full_ids = prefix_tokens + text_ids + suffix_tokens
        batch_input_ids.append(full_ids)

    # Padding
    max_len = max(len(ids) for ids in batch_input_ids)
    padded_input_ids = []
    attention_masks = []
    
    for ids in batch_input_ids:
        padding_len = max_len - len(ids)
        padded_ids = ids + [reranker_tokenizer.pad_token_id] * padding_len
        mask = [1] * len(ids) + [0] * padding_len
        padded_input_ids.append(padded_ids)
        attention_masks.append(mask)

    # 推論
    input_tensor = torch.tensor(padded_input_ids).to(reranker_model.device)
    mask_tensor = torch.tensor(attention_masks).to(reranker_model.device)

    outputs = reranker_model(input_ids=input_tensor, attention_mask=mask_tensor)
    batch_logits = outputs.logits
    
    # 取最後一個 Token 的 Logits
    last_token_indices = mask_tensor.sum(1) - 1
    target_logits = batch_logits[torch.arange(batch_logits.shape[0]), last_token_indices, :]

    # 計算 Yes 機率
    scores_pair = torch.stack([target_logits[:, token_false_id], target_logits[:, token_true_id]], dim=1)
    probs = torch.nn.functional.softmax(scores_pair, dim=1)
    return probs[:, 1].tolist() # Return Yes probability


# ==========================================
# 4. 混合檢索 + 重排序 (核心流程)
# ==========================================

def hybrid_search_rerank(query, top_n=20, top_k=3):
    # Step 1: Qdrant Hybrid Search (Vector + BM25)
    query_vector = embedding_model.encode([query])[0].tolist()
    
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model="Qdrant/bm25"),
                using="sparse",
                limit=top_n
            ),
            models.Prefetch(
                query=query_vector,
                using="dense",
                limit=top_n
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_n,
    )
    
    candidate_docs = [hit.payload['text'] for hit in search_result.points]
    if not candidate_docs:
        return ""

    # Step 2: Rerank (使用 Qwen3)
    scores = get_rerank_scores(query, candidate_docs)
    
    # 排序並取 Top K
    ranked_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in ranked_results[:top_k]]
    
    return "\n\n".join(top_docs)


# ==========================================
# 5. 資料處理與主程式
# ==========================================

def ingest_data():
    """建立資料庫並寫入資料"""
    files = glob.glob("data_*.txt")
    if not files:
        print("警告: 找不到 data_*.txt，請確認檔案位置")
        return

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}
    )
    
    documents = []
    # 讀檔與切塊
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            for i in range(0, len(text), 250):
                chunk = text[i:i+300]
                if len(chunk) > 50: documents.append(chunk)

    print(f"正在寫入 {len(documents)} 筆資料...")
    dense_vectors = embedding_model.encode(documents).tolist()
    
    points = []
    for idx, (doc, vector) in enumerate(zip(documents, dense_vectors)):
        points.append(models.PointStruct(
            id=idx,
            vector={
                "dense": vector,
                "sparse": models.Document(text=doc, model="Qdrant/bm25")
            },
            payload={"text": doc}
        ))
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("資料庫建立完成。")

def generate_answer(query, context):
    """使用 LLM 回答"""
    prompt = f"請根據以下資料回答問題，若無答案請說不知道。\n\n[參考資料]\n{context}\n\n[問題]\n{query}"
    res = llm_client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

def main():
    # 建立資料庫 (如果已建立過可註解掉)
    ingest_data()
    
    input_csv = "Re_Write_questions.csv"
    output_csv = "CW04_Answer.csv"
    
    if not os.path.exists(input_csv):
        print(f"錯誤：找不到 {input_csv}")
        return

    df = pd.read_csv(input_csv)
    
    # 自動抓取問題欄位
    possible_cols = ['question', 'questions', '題目']
    q_col = next((c for c in possible_cols if c in df.columns), df.columns[0])
    print(f"使用欄位 '{q_col}' 作為問題來源")

    results = []
    print(f"開始處理 {len(df)} 題...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        q = str(row[q_col])
        
        # 1. 檢索 + 重排
        context = hybrid_search_rerank(q)
        
        # 2. 生成回答
        ans = generate_answer(q, context)
        
        results.append({
            "question": q,
            "answer": ans,
            "context_preview": context[:50] + "..."
        })
        
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"完成！結果已存至 {output_csv}")

if __name__ == "__main__":
    main()