import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 初始化模型
model = ChatOpenAI(
    model="google/gemma-3-27b-it",
    openai_api_base="https://ws-02.wade0426.me/v1",
    openai_api_key="your_api_key", 
    temperature=0,
    max_tokens=40
)

# 定義 Prompt 範本
ig_prompt = ChatPromptTemplate.from_template("你是一位 IG 網紅，請針對主題「{topic}」寫一個 30 字以內的超短貼文，多放點表情符號。")
linkedin_prompt = ChatPromptTemplate.from_template("你是一位 LinkedIn 專家，請針對主題「{topic}」寫一句 30 字內的專業精簡觀點。")

# 建立平行處理鏈
chain = RunnableParallel(
    instagram=ig_prompt | model | StrOutputParser(),
    linkedin=linkedin_prompt | model | StrOutputParser()
)

topic_input = input("輸入主題: ")

# --- 1. 流式輸出 (Streaming) ---
print("\n--- 流式輸出 ---")
for chunk in chain.stream({"topic": topic_input}):
    print(chunk)

# --- 2. 批次處理 (Batch) ---
print("\n--- 批次處理 ---")
start_time = time.time()
result = chain.invoke({"topic": topic_input})
end_time = time.time()

# 輸出結果
print(f"耗時: {end_time - start_time:.2f} 秒")
print("-" * 30)
print(f"【 LinkedIn 專家說 】：\n{result['linkedin']}\n")
print("-" * 30)
print(f"【 IG 網紅說 】：\n{result['instagram']}")