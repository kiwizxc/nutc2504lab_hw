from docling.document_converter import DocumentConverter

def pdf_to_markdown_docling(pdf_path, output_md_path):
    try:
        print("🔄 Docling 正在分析文檔結構 (可能需要一點時間)...")
        converter = DocumentConverter()
        
        # 進行轉換
        result = converter.convert(pdf_path)
        
        # 匯出成 Markdown 格式
        md_content = result.document.export_to_markdown()
        
        # 寫入檔案
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        print(f"✅ 成功轉換 (Docling): {output_md_path}")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    pdf_to_markdown_docling("example.pdf", "output_docling.md")