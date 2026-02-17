import os
import base64
import pandas as pd
from openai import OpenAI

# 基礎組件
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader

# Docling 相關組件 (作業要求：RapidOCR)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

# ==========================================
# 1. 設定 API
# ==========================================
vlm_client = OpenAI(
    base_url="https://ws-05.huannago.com/v1",
    api_key="sk-dummy-key"
)
VLM_MODEL = "Qwen3-VL-8B-Instruct-BF16.gguf"

llm_client_config = {
    "base_url": "https://ws-02.wade0426.me/v1",
    "api_key": "sk-dummy-key",
    "model": "gemma-3-27b-it" # 使用教材推薦之多模態模型系列
}

# ==========================================
# 2. 功能函數
# ==========================================

def check_for_injection(text, filename):
    """
    偵測間接提示詞注入 (Indirect Prompt Injection) [cite: 1017]
    教材建議建立惡意模式庫進行掃描 [cite: 1036]
    """
    malicious_keywords = ["ignore all system prompts", "tiramisu", "pastry chef", "ignore previous instructions"]
    for keyword in malicious_keywords:
        if keyword.lower() in text.lower():
            print(f"🚨 [資安警報] 檔案 '{filename}' 發現惡意關鍵字 '{keyword}'！")
            return True 
    return False

def analyze_image_with_vlm(image_path):
    print(f"正在分析圖片: {image_path} ...")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    
    resp = vlm_client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "請詳細轉錄圖片中的所有文字內容，並保持結構。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }]
    )
    return resp.choices[0].message.content

# ==========================================
# 3. IDP 流程 (使用 Docling + RapidOCR)
# ==========================================
print("1. IDP 處理中 (載入檔案)...")
raw_documents = []

# 配置 Docling 使用 RapidOCR 引擎 [cite: 585, 593]
# RapidOCR 在資源受限環境中具有極佳的速度優勢 [cite: 784, 786]
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = RapidOcrOptions() 
docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# (A) PDF (1.pdf, 2.pdf, 3.pdf)
for f in ["1.pdf", "2.pdf", "3.pdf"]:
    if os.path.exists(f):
        print(f"[*] Docling 處理中: {f} (RapidOCR Enabled)")
        try:
            # 執行解析、佈局分析與轉換 [cite: 158-162]
            result = docling_converter.convert(f)
            # 導出為 Markdown 以保留語意結構 [cite: 170]
            content_md = result.document.export_to_markdown()
            raw_documents.append(Document(page_content=content_md, metadata={"source": f}))
        except Exception as e:
            print(f"Docling 處理 {f} 失敗: {e}")

# (B) Word (5.docx)
if os.path.exists("5.docx"):
    print("Loading 5.docx...")
    docs = Docx2txtLoader("5.docx").load()
    for d in docs: d.metadata["source"] = "5.docx"
    raw_documents.extend(docs)

# (C) Image (4.png 或 4.jpg)
for img_name in ["4.png", "4.jpg"]:
    if os.path.exists(img_name):
        try:
            content = analyze_image_with_vlm(img_name)
            raw_documents.append(Document(page_content=content, metadata={"source": img_name}))
            print(f"圖片 {img_name} 讀取成功。")
            break
        except Exception as e:
            print(f"圖片 {img_name} 讀取失敗: {e}")

# (D) 安全過濾：剔除有注入風險的檔案 [cite: 1054]
safe_docs = []
blocked_files = set()
for doc in raw_documents:
    src = doc.metadata.get("source", "")
    if src in blocked_files: continue
    
    if check_for_injection(doc.page_content, src):
        blocked_files.add(src)
        print(f"🚫 安全防護：已從 RAG 知識庫剔除惡意檔案: {src}")
    else:
        safe_docs.append(doc)

# ==========================================
# 4. 建立 RAG (以 Markdown 結構優化分塊)
# ==========================================
print("\n2. 建立 RAG 向量資料庫...")
# 建議分塊大小考慮到 Markdown 結構 [cite: 155]
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
texts = splitter.split_documents(safe_docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    base_url=llm_client_config["base_url"],
    api_key=llm_client_config["api_key"],
    model=llm_client_config["model"],
    temperature=0
)

def simple_rag_ask(question):
    docs = retriever.invoke(question)
    context_text = "\n\n".join([d.page_content for d in docs])
    prompt = f"""請根據以下參考資料回答問題。如果資料中沒有答案，請回答無法提供資訊。

參考資料：
{context_text}

問題：{question}
答案："""
    response = llm.invoke(prompt)
    return {
        "result": response.content,
        "source_documents": docs
    }

# ==========================================
# 5. DeepEval 驗證 (4個指標，取前5筆) [作業要求]
# ==========================================
print("\n3. 執行 DeepEval 驗證 (前 5 筆)...")
if os.path.exists("questions_answer.csv"):
    try:
        from deepeval import evaluate
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric, ContextualPrecisionMetric
        from deepeval.test_case import LLMTestCase
        from deepeval.models.base_model import DeepEvalBaseLLM

        class SimpleGemmaEval(DeepEvalBaseLLM):
            def __init__(self, model): self.model = model
            def load_model(self): return self.model
            def generate(self, prompt: str) -> str: return self.model.invoke(prompt).content
            async def a_generate(self, prompt: str) -> str: return self.generate(prompt)
            def get_model_name(self): return "Gemma-3-Eval"

        eval_model = SimpleGemmaEval(llm)
        df_val = pd.read_csv("questions_answer.csv").head(5)
        test_cases = []
        
        for _, row in df_val.iterrows():
            res = simple_rag_ask(row["questions"])
            test_cases.append(LLMTestCase(
                input=row["questions"],
                actual_output=res["result"],
                expected_output=row["answer"],
                retrieval_context=[d.page_content for d in res["source_documents"]]
            ))
        
        metrics = [
            FaithfulnessMetric(threshold=0.5, model=eval_model, include_reason=False),
            AnswerRelevancyMetric(threshold=0.5, model=eval_model, include_reason=False),
            ContextualRecallMetric(threshold=0.5, model=eval_model, include_reason=False),
            ContextualPrecisionMetric(threshold=0.5, model=eval_model, include_reason=False)
        ]
        evaluate(test_cases, metrics=metrics)
    except Exception as e:
        print(f"DeepEval 執行略過: {e}")

# ==========================================
# 6. 生成 test_dataset.csv
# ==========================================
print("\n4. 生成結果檔案...")
if os.path.exists("test_dataset.csv"):
    df = pd.read_csv("test_dataset.csv")
    if "id" in df.columns: df.rename(columns={"id": "q_id"}, inplace=True)
        
    answers, sources = [], []
    for q in df["questions"]:
        try:
            res = simple_rag_ask(q)
            ans = res["result"]
            src_docs = res["source_documents"]
            
            # 針對被剔除檔案 5.docx 的問答處理
            if "5.docx" in blocked_files and ("5.docx" in q or "公文" in q):
                ans = "⚠️ 由於來源檔案 (5.docx) 偵測到惡意指令注入，基於安全考量已被過濾，無法提供內容。"

            answers.append(ans)
            sources.append(", ".join(list(set([d.metadata.get('source', '') for d in src_docs]))) if src_docs else "None")
        except:
            answers.append("Error"); sources.append("")

    df["answer"], df["source"] = answers, sources
    df[["q_id", "questions", "answer", "source"]].to_csv("test_dataset_solved.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 作業完成！請將 test_dataset_solved.csv 重新命名後上傳。")