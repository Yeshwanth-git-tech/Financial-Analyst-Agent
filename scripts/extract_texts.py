import os
import pdfplumber

INPUT_DIR = "data/sec_filings"
OUTPUT_DIR = "data/extracted_txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if file.endswith(".pdf"):
        filepath = os.path.join(INPUT_DIR, file)
        with pdfplumber.open(filepath) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        output_file = os.path.join(OUTPUT_DIR, file.replace(".pdf", ".txt"))
        with open(output_file, "w") as out:
            out.write(text)
        print(f"✅ Saved extracted text to: {output_file}")