from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ApiVlmOptions
from docling.datamodel.settings import ResponseFormat
import os

# --- 設定 OLM OCR 2 的函式 (整合自雲端範例) ---
def olmocr2_vlm_options(
    model: str = "allenai/olmOCR-2-7B-1025-FP8",
    hostname_and_port: str = "ws-01.wade0426.me/v1", # 注意這裡不需要 https://，因為下面會自動加
    prompt: str = "Convert this page to markdown.",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key: str = "EMPTY", # 雲端範例通常不需要 key，或填入你的 key
) -> ApiVlmOptions:

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
   
    options = ApiVlmOptions(
        url=f"http://{hostname_and_port}/chat/completions", # 使用 http 連線到工作坊伺服器
        params=dict(
            model=model,
            max_tokens=max_tokens,
        ),
        headers=headers,
        prompt=prompt,
        timeout=120,  # VLM 處理時間較長
        scale=2.0,    # 圖片放大以提升細節識別
        temperature=temperature,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options

# --- 主程式 ---

input_filename = "sample_table.pdf"
output_filename = "output_olm_ocr.md"

if not os.path.exists(input_filename):
    print(f"❌ 找不到檔案: {input_filename}")
    exit()

# 設定 Pipeline: 啟用 OCR 並指定使用 VLM (OLM OCR 2)
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True

# 關鍵：將 OCR 選項切換為 VLM，並載入自定義設定
pipeline_options.ocr_options = olmocr2_vlm_options()

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

print(f"🚀 正在使用 OLM OCR 2 (雲端 GPU) 轉換 {input_filename} ...")
print("這可能需要一點時間，請耐心等待...")

result = converter.convert(input_filename)

markdown_content = result.document.export_to_markdown()
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"✅ 轉換完成！深度解析結果已儲存為 {output_filename}")