import os
import json
from urllib.parse import unquote  # 引入解碼工具
from typing import TypedDict, Annotated, Literal, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, add_messages

# 匯入工具模組 (請確保 search_searxng.py 和 vlm_read_website.py 在同目錄下)
from search_searxng import search_searxng
from vlm_read_website import vlm_read_website

# ============ 配置區 ============
# 使用 ws-03 伺服器
llm = ChatOpenAI(
    base_url="https://ws-03.wade0426.me/v1",
    api_key="EMPTY",
    model="/models/gpt-oss-120b",
    temperature=0
)

CACHE_FILE = "verification_cache.json"

# ============ 1. 定義狀態 (State) ============
class GraphState(TypedDict):
    question: str
    knowledge_base: str
    messages: Annotated[list[BaseMessage], add_messages]
    loop_count: int
    is_cache_hit: bool
    final_answer: str
    visited_urls: List[str] # 記錄已讀過的網址

# ============ 2. 快取工具 ============
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {}
    return {}

def save_cache(question, answer):
    data = load_cache()
    data[question] = answer
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ============ 3. 定義節點 (Nodes) ============

def check_cache_node(state: GraphState):
    print("\n--- 1. 檢查快取 (Cache Check) ---")
    data = load_cache()
    
    if state["question"] in data:
        print("✅ 命中快取！準備直接輸出。")
        return {
            "is_cache_hit": True, 
            "final_answer": data[state["question"]],
            "knowledge_base": "",
            "loop_count": 0,
            "visited_urls": []
        }
    
    print("❌ 未命中，進入查證流程。")
    return {
        "is_cache_hit": False, 
        "knowledge_base": "", 
        "loop_count": 0,
        "visited_urls": []
    }

# 【主要修改處】Planner 現在會印出理由
def planner_node(state: GraphState):
    print(f"\n--- 2. 決策中 (Planner) [Loop: {state['loop_count']}] ---")
    
    if state["loop_count"] >= 4:
        print("⚠️ 已達最大搜尋次數，強制進行回答。")
        return {"messages": [AIMessage(content="Planner決策: ENOUGH")]}

    # 修改 Prompt：要求 AI 先給理由，再給決策
    prompt = f"""
    使用者問題: {state['question']}
    
    目前已收集的外部資訊(Knowledge Base):
    {state.get('knowledge_base', '尚無資訊')}
    
    請判斷：目前的資訊是否已經「足夠」回答使用者的問題？
    
    請依照以下格式回覆：
    理由：(請簡短說明還缺少什麼關鍵數據、年份或是細節，或者為什麼資訊已足夠)
    決策：(最後一行請務必只輸出 "SEARCH" 或 "ENOUGH")
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    
    # 印出 AI 的思考過程
    print(f"🤔 判斷理由:\n{content}")

    # 解析邏輯 (抓取最後一行的決策)
    lines = content.split('\n')
    last_line = lines[-1].upper()
    
    # 簡單的防呆判斷，如果最後一行包含 SEARCH 或 ENOUGH 就抓取
    if "SEARCH" in last_line or "SEARCH" in content.split("決策：")[-1]:
        decision = "SEARCH"
    else:
        decision = "ENOUGH"
    
    print(f"🤖 Planner 最終決定: {decision}")
    return {"messages": [AIMessage(content=f"Planner決策: {decision}")]}

def query_gen_node(state: GraphState):
    print("\n--- 3. 生成搜尋關鍵字 (Query Gen) ---")
    prompt = f"""
    使用者問題: {state['question']}
    目前已知資訊: {state.get('knowledge_base', '')}
    
    請生成一個「最適合搜尋引擎」的關鍵字，用來查找缺少的資訊。
    只回覆關鍵字本身，不要加任何標點符號。
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    query = response.content.strip()
    print(f"🔑 生成關鍵字: {query}")
    return {"messages": [AIMessage(content=query)]}

def search_and_read_node(state: GraphState):
    query = state["messages"][-1].content
    print(f"\n--- 4. 執行搜尋與閱讀 (Search & VLM) ---")
    
    results = search_searxng(query, limit=5)
    
    target_result = None
    # 將已訪問過的網址解碼
    visited_normalized = [unquote(u) for u in state.get("visited_urls", [])]
    
    if results:
        for res in results:
            res_url_norm = unquote(res['url'])
            
            if res_url_norm not in visited_normalized:
                target_result = res
                break
            else:
                print(f"🙈 跳過已讀過的網址: {res['title']}")
    
    new_info = ""
    new_visited_url = []
    
    if target_result:
        title = target_result['title']
        url = target_result['url']
        print(f"🌐 鎖定網頁: {title}")
        print(f"🔗 URL: {url}")
        
        print("📸 VLM 正在閱讀網頁 (請稍候)...")
        # 注意: 這裡會呼叫外部檔案
        content = vlm_read_website(url, title)
        
        new_info = f"\n=== 新增資料來源: {title} ===\n{content}\n"
        new_visited_url = [unquote(url)]
    else:
        print("⚠️ 搜尋結果皆已讀過，或無相關結果。")
        new_info = "\n[系統] 此關鍵字查無新資料，請嘗試其他方向。\n"

    return {
        "knowledge_base": state.get("knowledge_base", "") + new_info,
        "loop_count": state["loop_count"] + 1,
        "visited_urls": state.get("visited_urls", []) + new_visited_url
    }

def answer_node(state: GraphState):
    if state.get("is_cache_hit"):
        return {}

    print("\n--- 5. 生成最終回答 (Final Answer) ---")
    prompt = f"""
    使用者問題: {state['question']}
    
    這是你辛苦查證後收集到的資訊:
    {state['knowledge_base']}
    
    請根據上述資訊，完整且專業地回答使用者的問題。
    並在最後附上參考來源。
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    save_cache(state["question"], response.content)
    print("💾 已將結果寫入快取。")
    
    return {"final_answer": response.content}

# ============ 4. 定義路由 (Router) ============

def cache_router(state: GraphState) -> Literal["hit", "miss"]:
    if state.get("is_cache_hit"): return "hit"
    return "miss"

def planner_router(state: GraphState) -> Literal["answer", "query"]:
    last_msg = state["messages"][-1].content
    if "ENOUGH" in last_msg: return "answer"
    return "query"

# ============ 5. 組裝 Graph ============

workflow = StateGraph(GraphState)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_and_read_node)
workflow.add_node("final_answer", answer_node)

workflow.set_entry_point("check_cache")

workflow.add_conditional_edges(
    "check_cache",
    cache_router,
    {
        "miss": "planner",      
        "hit": "final_answer"   
    }
)

workflow.add_conditional_edges(
    "planner",
    planner_router,
    {
        "query": "query_gen",    
        "answer": "final_answer" 
    }
)

workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")
workflow.add_edge("final_answer", END)

app = workflow.compile()

# ============ 6. 執行區 ============
if __name__ == "__main__":
    
    # 嘗試繪製 ASCII 流程圖
    try:
        print(app.get_graph().draw_ascii())
    except Exception:
        pass

    print(f"🚀 自動查證 AI 已啟動！(Model: /models/gpt-oss-120b)")
    
    while True:
        try:
            q = input("\n請輸入想查證的問題 (q 離開): ").strip()
            
            if q.lower() == 'q': break
            
            if not q:
                print("⚠️ 請輸入有效的問題！")
                continue
            
            inputs = {"question": q, "messages": []}
            result = app.invoke(inputs)
            
            print("\n" + "="*30)
            print("💡 最終結果:")
            print(result["final_answer"])
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")