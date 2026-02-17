from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions
)
from docling.datamodel.base_models import InputFormat
import os

# 設定檔案路徑 (假設 pdf 在同一層目錄)
input_filename = "sample_table.pdf"
output_filename = "output_rapid.md"

# 檢查檔案是否存在
if not os.path.exists(input_filename):
    print(f"❌ 找不到檔案: {input_filename}，請確認檔案位置。")
    exit()

# 設定 Pipeline: 啟用 OCR 並指定使用 RapidOCR
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.ocr_options = RapidOcrOptions()

# 初始化轉換器
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

print(f"🚀 正在使用 RapidOCR 轉換 {input_filename} ...")
result = converter.convert(input_filename)

# 輸出 Markdown
markdown_content = result.document.export_to_markdown()
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"✅ 轉換完成！結果已儲存為 {output_filename}")