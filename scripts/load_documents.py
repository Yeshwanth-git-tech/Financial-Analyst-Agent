import os
from llama_index.core import Document

def extract_documents(input_dir, save_txt_dir=None):
    docs = []
    os.makedirs(save_txt_dir, exist_ok=True) if save_txt_dir else None

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)

        # Accept only HTML and Markdown
        if file.endswith((".html", ".htm", ".md", ".txt")):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            docs.append(Document(text=text, metadata={"filename": file}))

            if save_txt_dir:
                out_path = os.path.join(save_txt_dir, file.replace(".html", ".txt").replace(".md", ".txt"))
                with open(out_path, "w", encoding="utf-8") as out:
                    out.write(text)

    return docs