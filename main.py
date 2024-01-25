#!/usr/bin/env python
"""
Reads text files, and outputs statistics about linguistic structures that were pre-determined.
"""

import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from pathlib import Path
import spacy
import pickle
from collections import Counter

from timexy import Timexy

SPACY_MODEL = "en_core_web_lg"#"en_core_web_sm"
SPACY_MODEL_SUFFIX = SPACY_MODEL.split("_")[-1]


import spacy.cli
spacy.cli.download(SPACY_MODEL)


READ_PATH = <insert_folder_to_read_from> #Should be of type pathlib.Path
WRITE_PATH = <insert_folder_to_write_to> #Should be of type pathlib.Path

USE_CACHE = False 
CACHE_PATH = <insert_cache_path> #Optional, depending on USE_CACHE, Should be of type pathlib.Path

DO_UNIFIED = False

timexy_config = {
    "kb_id_type": "timex3",  # possible values: 'timex3'(default), 'timestamp'
    "label": "timexy",       # default: 'timexy'
    "overwrite": False       # default: False
}



nlp = spacy.load(SPACY_MODEL)
nlp.add_pipe("timexy", config=timexy_config, before="ner")

def get_cached_docs(full_pth: Path) -> Optional[List[spacy.tokens.Doc]]:
  '''
  Load cached docs
  :param full_pth: full path to pickled cached docs file
  :return: the content of the given file if exists, otherwise None
  '''
  if full_pth.is_file():
    with open(full_pth, 'rb') as pickleFile:
      docs = pickle.load(pickleFile)
      return docs
  return None

def get_token_repr(tokens: List[spacy.tokens.Token]) -> List[Dict[str, str]]:
  '''
  Returns a string representation of each token in the list. Each string is a concatenation of selected token attributes
  (e.g., lemma, pos)
  :param tokens: list of spacy tokens
  :return: a list of string representation of each token
  '''
  attrs = ["pos_", "text", "lemma_", "dep_", "tag_", "head", "morph", "ent_type_"]
  res = [{a.replace("_", ""): str(getattr(t, a, "")) for a in attrs} for t in tokens]
  return res

def handle_doc(doc: spacy.tokens.Doc, fname: str) -> Tuple[pd.DataFrame, int]:
  '''
  This function takes a spacy doc object and returns a pandas dataframe with the lingustic structures extracted from the
  doc, according to pre-determined rules (see body of function). It also returns the number of sentences in the doc
  :param doc: given spacy doc object to explore
  :param fname: the file name from which the doc was read from
  :return: a 2-tuple:
           1) dataframe with the lingusitc structures found
           2) count of the number of sentences in the doc
  '''
  res = []
  sentence_count = 0
  academic_text = "text_acad" in fname
  for i, s in enumerate(doc.sents):
    sentence_count += 1
    time_tokens = []
    if i > 0 and i % 500 == 0:
      print(f"{datetime.now()} sent #{i}")
    j=0
    verb_time_expression_past_in_sentence = False
    verb_time_expression_future_in_sentence = False
    while j<len(s):
      #Per token loop
      token = s[j]

      #For academic texts, we ignore everything inside parenthesis as to not "catch" irrelevant time expressions that are part of an academic citation
      if academic_text and token.text == "(":
        while token.text != ")" and j<len(s)-1:
          j+=1
          token = s[j]

      if token.ent_type_ and token.ent_type_ == 'DATE':
        time_tokens.append(token)

      # For spacy codes (e.g. 'JJS'), see: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
      if token.text == "'s" and token.head.dep_ == 'poss':
        res.append({"fname": fname, "sentence": s, "structure": "L1_possessive_marking", "data": get_token_repr([token.head, token])})
      elif token.lower_ == 'the' and j+1<len(s) and token.nbor(1).tag_ == 'JJS':
         res.append({"fname": fname, "sentence": s, "structure": "L1_superlative_definite_article_omission", "data": get_token_repr([token, token.nbor(1)])})
         res.append({"fname": fname, "sentence": s, "structure": "L2_superlative_form", "data": get_token_repr([token, token.nbor(1)])})
         j+=1
      elif token.lower_ == 'the' and j+2<len(s) and token.nbor(1).pos_ == 'ADV'  and token.nbor(2).tag_ == 'JJS':
         res.append({"fname": fname, "sentence": s, "structure": "L1_superlative_definite_article_omission", "data": get_token_repr([token, token.nbor(1), token.nbor(2)])})
         res.append({"fname": fname, "sentence": s, "structure": "L2_superlative_form", "data": get_token_repr([token, token.nbor(1), token.nbor(2)])})
         j+=2
      elif token.tag_ == 'JJR' and token.lemma_.lower() != 'more':
        res.append({"fname": fname, "sentence": s, "structure": "L2_comparative_form", "data": get_token_repr([token])})
      elif token.lemma_.lower() == 'more' and token.head.tag_ == 'JJ':
        res.append({"fname": fname, "sentence": s, "structure": "L2_comparative_more", "data": get_token_repr([token, token.head])})
      elif j+2<len(s) and token.lower_ == 'the' and token.nbor(1).lower_ == 'most' and token.nbor(2).pos_ =='ADJ':
         res.append({"fname": fname, "sentence": s, "structure": "L2_superlative_most", "data": get_token_repr([token, token.nbor(1), token.nbor(2)])})
         j+=2
      elif token.lemma_.lower() in ["i", "we"] and token.head.pos_ == 'VERB':
        res.append({"fname": fname, "sentence": s, "structure": "L1_L2_1st_person_prodrop", "data": get_token_repr([token, token.head])})
      elif token.pos_ == 'ADJ' and token.dep_ == 'acomp' and token.head.lemma_ == 'be':
        res.append({"fname": fname, "sentence": s, "structure": "L1_L2_copula_omission", "data": get_token_repr([token, token.head])})
      elif token.lemma_ in ['a', 'an'] and token.pos_ == 'DET' and token.dep_ == 'det':
        res.append({"fname": fname, "sentence": s, "structure": "L1_L2_indefinite_article_omission", "data": get_token_repr([token])})
      elif token.pos_ == 'VERB' and 'Tense=Past' in str(token.morph) and len(time_tokens)>0 and not verb_time_expression_past_in_sentence:
        #Note that morph is necessary here and can't do only with VBD etc. (e.g.: "I've done the homework already")
        #Note the said time expression and verb can be anywhere in the sentence
        res.append({"fname": fname, "sentence": s, "structure": "no_interference_verb_time_expression_past", "data": get_token_repr([token]+time_tokens)})
        verb_time_expression_past_in_sentence = True
      elif token.lemma_.lower() in ['will', 'shall'] and token.dep_ == 'aux' and token.head.tag_ == 'VB' and len(time_tokens)>0 and not verb_time_expression_future_in_sentence:
        #As for finding a verb in future tense, see: https://github.com/explosion/spaCy/discussions/2767#discussioncomment-186632
        res.append({"fname": fname, "sentence": s, "structure": "no_interference_verb_time_expression_future", "data": get_token_repr([token, token.head]+time_tokens)})
        verb_time_expression_future_in_sentence = True
      elif token.dep_ in ["nummod", "num", "number", "nummod", "quantmod"]  and token.head.pos_ == "NOUN" and token.head.tag_ == "NNS":
         res.append({"fname": fname, "sentence": s, "structure": "no_interference_quantifier_noun_plural_agreement", "data": get_token_repr([token.head, token])})
      j+=1


  return pd.DataFrame(res), sentence_count


print(f"{datetime.now()} Start")

sentence_count_overall = 0
for pth_i, pth in enumerate(READ_PATH.glob('*.txt')):
  print(f"{datetime.now()} {pth} {pth_i}")
  analyzied_doc_dfs = []
  out_file_counter = 0
  with open(pth, errors="ignore") as file:
    parsed_docs_path = Path(CACHE_PATH, f"{pth.stem}.pkl")
    if USE_CACHE and (cached_docs := get_cached_docs(parsed_docs_path)):
      for cached_doc in cached_docs:
        res_df, sentence_count = handle_doc(cached_doc, pth.stem)
        analyzied_doc_dfs.append(res_df)
        sentence_count_overall += sentence_count

    else:
      parsed_docs = []
      i = 0
      while line := file.readline(): 
          doc = nlp(line.strip())
          if USE_CACHE:
            parsed_docs.append(doc)
          res_df, sentence_count = handle_doc(doc, pth.stem)
          analyzied_doc_dfs.append(res_df)
          sentence_count_overall += sentence_count
          i+=1
          if i%100 == 0:
            out_fname = f"{pth.stem}_{out_file_counter}.csv"
            print(f"{datetime.now()} doc#{i}, writing to file {out_fname}")
            pd.concat(analyzied_doc_dfs, ignore_index=True).to_csv(
              Path(WRITE_PATH, out_fname), index=False)
            analyzied_doc_dfs = []
            out_file_counter += 1

      if USE_CACHE:
        with open(parsed_docs_path, 'wb') as f:
          pickle.dump(parsed_docs, f)
    if len(analyzied_doc_dfs) > 0:
      pd.concat(analyzied_doc_dfs, ignore_index=True).to_csv(Path(WRITE_PATH, f"{pth.stem}_{out_file_counter}.csv"), index=False)

print(f"{datetime.now()} Number of sentences found: {sentence_count_overall}")

if DO_UNIFIED:
  ###### Create unified file ######
  dfs = []
  for pth_i, pth in enumerate(READ_PATH.glob('*.csv')):
    print(datetime.now(), pth_i, pth)
    df = pd.read_csv(pth)
    if len(df.index>0):
      dfs.append(df)

  unified_df = pd.concat(dfs, ignore_index=True)
  unified_df.to_csv(Path(WRITE_PATH, "unified_data.csv"), index = False)

print(f"{datetime.now()} FIN")
