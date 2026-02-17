import sys
import os
import time
import requests
import re
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# ==========================================
# 1. 環境與 API 設定 (整合您的新程式碼)
# ==========================================
sys.stdout.reconfigure(encoding='utf-8')

# ASR API 設定 (來自您提供的範例)
ASR_BASE = "https://3090api.huannago.com"
CREATE_URL = f"{ASR_BASE}/api/v1/subtitle/tasks"
# 這是您提供的帳密
ASR_AUTH = ("nutc2504", "nutc2504") 

# 課程 LLM 伺服器設定 (ws-03)
VLLM_BASE_URL = "https://ws-03.wade0426.me/v1"
VLLM_API_KEY = "vllm-token"
MODEL_NAME = "/models/gpt-oss-120b"

# 音檔名稱
AUDIO_FILE = "Podcast_EP14_30s.wav"

# 備用逐字稿
BACKUP_TRANSCRIPT_TABLE = """
| 時間 | 內容 |
| :--- | :--- |
| 00:00:00 | (備用資料) 歡迎來到天下文化Podcast... |
"""

# ==========================================
# 2. 定義 State
# ==========================================
class MeetingState(TypedDict):
    raw_transcript: str       # 純文字 (給 AI 讀)
    formatted_transcript: str # 表格 (給人類看)
    detailed_minutes: str
    summary: str
    final_report: str

# ==========================================
# 3. 輔助函式：SRT 轉 Markdown 表格
# ==========================================
def srt_to_markdown_table(srt_text):
    """將 SRT 字幕格式轉換為 Markdown 表格"""
    try:
        lines = srt_text.strip().split('\n')
        md_table = "| 時間 | 發言內容 |\n| :--- | :--- |\n"
        
        # 簡單的狀態機解析
        current_time = ""
        current_text = []
        
        for line in lines:
            line = line.strip()
            # 判斷是否為時間軸 (e.g., 00:00:00,000 --> 00:00:02,000)
            if '-->' in line:
                current_time = line.replace(',', '.') # 將逗號換成點，美觀一點
            # 判斷是否為純數字 (序號)，跳過
            elif line.isdigit() and not current_time: 
                continue
            # 空行代表一段結束
            elif line == "":
                if current_time and current_text:
                    text_content = " ".join(current_text)
                    md_table += f"| {current_time} | {text_content} |\n"
                    current_text = []
                    current_time = ""
            # 其他就是字幕內容
            else:
                if current_time: # 確保已經抓到時間了
                    current_text.append(line)
        
        # 處理最後一段 (如果沒有空行結尾)
        if current_time and current_text:
            text_content = " ".join(current_text)
            md_table += f"| {current_time} | {text_content} |\n"
            
        return md_table
    except Exception as e:
        return f"SRT 解析失敗: {e}\n原始內容:\n{srt_text}"

# ==========================================
# 4. 初始化 Client
# ==========================================
llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
    model=MODEL_NAME,
    temperature=0
)

# ==========================================
# 5. 定義節點 (更新 ASR 邏輯)
# ==========================================

# Node 1: ASR (使用您提供的正確 API 邏輯)
def asr_node(state: MeetingState):
    print(f"\n--- [ASR] 讀取音檔 & 上傳中... ---")
    
    if not os.path.exists(AUDIO_FILE):
        return {"raw_transcript": "無內容", "formatted_transcript": BACKUP_TRANSCRIPT_TABLE}

    try:
        # Step 1: 建立任務
        with open(AUDIO_FILE, 'rb') as f:
            # 使用您提供的 auth
            response = requests.post(
                CREATE_URL, 
                files={'audio': f}, 
                auth=ASR_AUTH, 
                timeout=60
            )
        
        if response.status_code != 200:
            raise Exception(f"上傳失敗: {response.text}")
            
        task_id = response.json().get('id')
        print(f"✅ 任務建立成功！ID: {task_id}")
        
        # 設定下載網址
        txt_url = f"{ASR_BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT"
        srt_url = f"{ASR_BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"
        
        # Step 2: 輪詢等待 TXT (純文字)
        print("⏳ [ASR] 等待轉錄結果 (TXT)...")
        txt_content = None
        
        # 等待 60 次 * 2 秒 = 120 秒
        for i in range(60):
            try:
                resp = requests.get(txt_url, timeout=10, auth=ASR_AUTH)
                if resp.status_code == 200:
                    txt_content = resp.text
                    print("✅ 取得純文字稿！")
                    break
            except:
                pass
            time.sleep(2)
            
        if not txt_content:
            raise Exception("等待 TXT 超時")

        # Step 3: 輪詢等待 SRT (時間軸)
        print("⏳ [ASR] 取得時間軸格式 (SRT)...")
        srt_content = None
        try:
            # 通常 TXT 好了 SRT 也差不多了，試幾次就好
            for i in range(5):
                resp = requests.get(srt_url, timeout=10, auth=ASR_AUTH)
                if resp.status_code == 200:
                    srt_content = resp.text
                    print("✅ 取得 SRT 時間軸！")
                    break
                time.sleep(1)
        except:
            print("⚠️ 無法取得 SRT，將使用純文字代替表格")

        # Step 4: 格式化輸出
        if srt_content:
            formatted_table = srt_to_markdown_table(srt_content)
        else:
            formatted_table = f"無法取得時間軸，原始內容：\n{txt_content}"

        return {
            "raw_transcript": txt_content,         # 給 AI 讀
            "formatted_transcript": formatted_table # 給人類看
        }

    except Exception as e:
        print(f"⚠️ [ASR] 發生錯誤: {e}")
        return {
            "formatted_transcript": BACKUP_TRANSCRIPT_TABLE,
            "raw_transcript": "轉錄失敗，使用備用資料。"
        }

# Node 2: Minutes Taker
def minutes_taker_node(state: MeetingState):
    print("--- [Minutes Taker] 整理記錄中... ---")
    prompt = ChatPromptTemplate.from_template("請將以下內容整理成 3 點關鍵紀錄：\n{text}")
    chain = prompt | llm | StrOutputParser()
    return {"detailed_minutes": chain.invoke({"text": state["raw_transcript"]})}

# Node 3: Summarizer
def summarizer_node(state: MeetingState):
    print("--- [Summarizer] 生成摘要中... ---")
    prompt = ChatPromptTemplate.from_template("請用一句話總結主旨：\n{text}")
    chain = prompt | llm | StrOutputParser()
    return {"summary": chain.invoke({"text": state["raw_transcript"]})}

# Node 4: Writer
def writer_node(state: MeetingState):
    print("--- [Writer] 撰寫報告中... ---")
    prompt = ChatPromptTemplate.from_template(
        """請根據以下資料，寫一份 Markdown 格式的聽書筆記：
        
        # 🎧 Podcast 聽書筆記
        
        ## 💡 一句話總結
        {summary}
        
        ## 📝 重點整理
        {details}
        
        ---
        ## 📜 詳細逐字稿 (Verbatim Transcript)
        {transcript}
        """
    )
    chain = prompt | llm | StrOutputParser()
    return {"final_report": chain.invoke({
        "summary": state["summary"], 
        "details": state["detailed_minutes"],
        "transcript": state["formatted_transcript"]
    })}

# ==========================================
# 6. 組裝 Graph (平行模式)
# ==========================================
workflow = StateGraph(MeetingState)

workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_taker_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("asr")
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

# ==========================================
# 7. 執行 (包含畫圖功能)
# ==========================================
if __name__ == "__main__":
    print(f"=== 開始執行智慧會議助手 (整合標準 API 版) ===")
    
    # 畫 ASCII 流程圖
    try:
        print(app.get_graph().draw_ascii())
    except:
        pass

    try:
        result = app.invoke({})
        print("\n" + "="*30)
        print("🎉 最終報告 (FINAL REPORT)")
        print("="*30)
        print(result["final_report"])
    except Exception as e:
        print(f"\n❌ 執行發生錯誤: {e}")