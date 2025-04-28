import json
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


def split_references(text):
    pattern = re.compile(r'(?s)(\[\d+\].*?)(?=(?:\[\d+\])|$)')
    refs = pattern.findall(text)
    return [clean_text(r.strip()) for r in refs]


def process_pdfs(input_dir: str, out_dir: str):
    corpus = []
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".pdf"):
            continue

        path = os.path.join(input_dir, fname)
        print(f"Processing: {path}")

        language = path.split('/')[-2].replace('.pdf', '')

        text = pymupdf4llm.to_markdown(path)

        reader = PdfReader(path)

        info = dict(reader.metadata)

        abstract = text.split('**ABSTRACT**')[1].split('**KEYWORDS**')[0].strip()

        #extract and separate the references
        raw_refs = text.split('**REFERENCES**')[1].strip()
        references = split_references(raw_refs)

        article = {
            "titulo": info['/Title'],
            "informacoes_url": "",
            "idioma": language,
            "storage_key": path,
            "author": info['/Author'],
            "data_publicacao": info['/CreationDate'],
            "resumo": abstract,
            "keywords": info['/Keywords'],
            "referencias": references,
            "text": text,
            "artigo_tokenizado": "",
            "pos_tagger": "",
            "lema": "",
        }

        corpus.append(article)

    with open(f"{out_dir}/corpus.json", "w", encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print('Finished processing')
