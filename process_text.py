import json
import os
import re
from datetime import datetime
import spacy
import pymupdf4llm
from PyPDF2 import PdfReader

language = {
    "english": "en_core_web_sm",
    "portuguese": "pt_core_news_sm"
}


def clean_ref(text):
    '''
    Remove: references style[1], \n
    :param text:
    :return:
    '''
    text = text.replace('\n', ' ')
    text = text.replace('*', '')
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

    text = re.sub(r'^#+\s*', '', text)


    return text


def process_pdfs(list_input_dir: list, out_dir: str):
    corpus = []


    for input_dir in list_input_dir:
        for fname in os.listdir(input_dir):
            if not fname.lower().endswith(".pdf"):
                continue

            path = os.path.join(input_dir, fname)
            print(f"Processing: {path}")

            lang = path.split('/')[-2].replace('.pdf', '')

            text = pymupdf4llm.to_markdown(path)

            # PYPDF provides metadata
            reader = PdfReader(path)
            info = dict(reader.metadata)

            abstract = text.split('ABSTRACT')[1].split('KEYWORDS')[0].strip()

            # extract and separate the references
            pattern = r'REFERENCES|REFERÊNCIAS'
            parts = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                raw_refs = parts[1].strip()
            else:
                raise ValueError("References section not found")

            raw_refs = raw_refs.split('\n###', 1)[0].strip()

            references = split_references(raw_refs)

            # remove references and clean the full text
            head, *_ = re.split(pattern, text, maxsplit=1)
            text = clean_full_text(head.strip())

            article = {
                "titulo": info['/Title'],
                "informacoes_url": "",
                "idioma": lang,
                "storage_key": path,
                "autores": info['/Author'].split(';'),
                "data_publicacao": format_date(str(info['/CreationDate'])),
                "resumo": clean_abstract(abstract),
                "keywords": info['/Keywords'],
                "referencias": [clean_ref(ref) for ref in references],
                "text": text,
                "artigo_tokenizado": tokenize_text(text, lang),
                "pos_tagger": pos_tag_text(text, lang),
                "lema": get_lemmas(text, lang),
            }
            corpus.append(article)



    with open(f"{out_dir}/corpus.json", "w", encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print('Finished processing')
