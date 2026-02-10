import pdfplumber

def pdf_to_markdown_plumber(pdf_path, output_md_path):
    # 建立 Markdown 內容的緩衝區
    md_content = f"# PDF Extract: {pdf_path}\n\n"
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # 提取該頁文字
                text = page.extract_text()
                
                if text:
                    # 加入頁碼標記 (方便閱讀)
                    md_content += f"## Page {i+1}\n\n"
                    # 將文字寫入，並保留基本換行
                    md_content += text + "\n\n---\n\n"
        
        # 寫入 .md 檔案
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        print(f"✅ 成功轉換 (pdfplumber): {output_md_path}")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    # 請確認 example.pdf 在同一目錄下
    pdf_to_markdown_plumber("example.pdf", "output_pdfplumber.md")