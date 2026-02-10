from markitdown import MarkItDown

def pdf_to_markdown_microsoft(pdf_path, output_md_path):
    try:
        # 初始化工具
        md = MarkItDown()
        
        # 轉換檔案
        result = md.convert(pdf_path)
        
        # 取得文字內容 (text_content)
        md_content = result.text_content
        
        # 寫入檔案
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        print(f"✅ 成功轉換 (Markitdown): {output_md_path}")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    pdf_to_markdown_microsoft("example.pdf", "output_markitdown.md")