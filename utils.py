import os
import pickle
import re

import pymupdf4llm
from PyPDF2 import PdfReader


def clean_text(text):
    '''
    Remove: references style[1], \n
    :param text:
    :return:
    '''
    pattern = r'\d*https?://\S+|[\w.+-]+@[\w-]+\.[\w.-]+'
    text = text.replace("\u200b", "")
    text = text.replace('- ', '')
    text = text.replace(' -', '')
    text = text.replace('\n', ' ')
    text = re.sub(r'([A-Za-z])-\s+([A-Za-z])', r'\1\2', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    text = re.sub(r'\d*https?://\S+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(pattern, '', text)
    # text = re.sub(r'(?:\*\*REFER[EÊ]NCIAS\*\*?|\*\*REFERENCES\*\*).*$', '', text, flags=re.DOTALL).strip()
    # pt-br = re.sub(r'\d+', '', pt-br)
    return text


def remove_references(text):
    pass
    return text

def process_pdfs(input_dir: str, out_dir: str):
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".pdf"):
            continue

        path = os.path.join(input_dir, fname)
        print(f"Processing: {path}")

        name = path.split('/')[-1].replace('.pdf', '')

        text = pymupdf4llm.to_markdown(path)

        with open(f"{out_dir}/{name}.txt", "w", encoding='utf-8') as f:
            f.write(text)

    print('Finished processing')