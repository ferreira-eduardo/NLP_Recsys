import json
import os
import pickle
import re
from datetime import datetime
import spacy
import pymupdf4llm
from PyPDF2 import PdfReader

language = {
    "english": "en_core_web_sm",
    "portuguese": "pt_core_news_sm"
}


def clean_text(text):
    '''
    Remove: references style[1], \n
    :param text:
    :return:
    '''
    text = text.replace('\n', ' ')
    text = text.replace('*', '')
    # text = text.replace('\n', ' ')
    # text = re.sub(r'([A-Za-z])-\s+([A-Za-z])', r'\1\2', text)
    # text = re.sub(r'\[[^\]]+\]', '', text)
    # text = re.sub(r'\d*https?://\S+', '', text)
    # text = re.sub(r'https?://\S+', '', text)
    # text = re.sub(pattern, '', text)
    return text


def split_references(text):
    pattern = re.compile(r'(?s)(\[\d+\].*?)(?=(?:\[\d+\])|$)')
    refs = pattern.findall(text)

    return [r.strip() for r in refs]


def format_date(date_str: str) -> str:
    clean = date_str.lstrip("D:")[:14]
    dt = datetime.strptime(clean, "%Y%m%d%H%M%S")
    return dt.strftime("%d-%m-%Y")


def clean_abstract(abstract: str) -> str:
    if '\n\n\n' in abstract:
        text = abstract.split('\n\n\n', 1)[1]
    else:
        text = abstract

    return text.replace('\n', ' ').strip()


def tokenize_text(text: str, lang: str) -> list:
    nlp = spacy.load(language[lang])

    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens


def get_lemmas(text: str, lang: str) -> list[str]:
    nlp = spacy.load(language[lang])
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]


def pos_tag_text(text: str, lang: str) -> list[tuple]:
    nlp = spacy.load(language[lang])
    doc = nlp(text)
    return [(tok.text, tok.pos_, tok.tag_) for tok in doc]


def clean_full_text(text: str) -> str:
    if '**1' in text:
        text = text.split('**1')[1]
    else:
        text = text
    text = re.sub(r'\n+', ' ', text).strip()
    text = text.replace("*", "")
    text = re.sub(r'\[\s*([^\[]+?)\s*\]', r'[\1]', text)

    return text


def process_pdfs(input_dir: str, out_dir: str):
    corpus = []
    i = 0
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".pdf"):
            continue

        path = os.path.join(input_dir, fname)
        print(f"Processing: {path}")

        lang = path.split('/')[-2].replace('.pdf', '')

        text = pymupdf4llm.to_markdown(path)

        reader = PdfReader(path)

        info = dict(reader.metadata)

        abstract = text.split('**ABSTRACT**')[1].split('**KEYWORDS**')[0].strip()

        # extract and separate the references
        raw_refs = text.split('**REFERENCES**')[1].strip()
        references = split_references(raw_refs)

        text = clean_full_text(text)

        article = {
            "titulo": info['/Title'],
            "informacoes_url": "",
            "idioma": lang,
            "storage_key": path,
            "author": info['/Author'],
            "data_publicacao": format_date(str(info['/CreationDate'])),
            "resumo": clean_abstract(abstract),
            "keywords": info['/Keywords'],
            "referencias": [clean_text(ref) for ref in references],
            "text": text,
            "artigo_tokenizado": tokenize_text(text, lang),
            "pos_tagger": pos_tag_text(text, lang),
            "lema": get_lemmas(text, lang),
        }
        i += 1
        corpus.append(article)
        if i == 2:
            break

    with open(f"{out_dir}/corpus.json", "w", encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print('Finished processing')
