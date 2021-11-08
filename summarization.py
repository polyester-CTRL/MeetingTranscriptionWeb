from janome.analyzer import Analyzer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenizer import Tokenizer as JanomeTokenizer  # sumyのTokenizerと名前が被るため
from janome.tokenfilter import POSKeepFilter, ExtractAttributeFilter
import re
import emoji
import mojimoji
import sys
import spacy
nlp = spacy.load('ja_ginza')

import neologdn

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words
from ginza import *

import sys
sys.path.append('../')
import UI


# アルゴリズム
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer


class JapaneseCorpus:
    # 
    def __init__(self):
        self.nlp = spacy.load('ja_ginza')
        self.analyzer = Analyzer(
            char_filters=[UnicodeNormalizeCharFilter(), RegexReplaceCharFilter(r'[(\)「」、。]', ' ')],  # ()「」、。は全てスペースに置き換える
            tokenizer=JanomeTokenizer(),
            token_filters=[POSKeepFilter(['名詞', '形容詞', '副詞', '動詞']), ExtractAttributeFilter('base_form')]  # 名詞・形容詞・副詞・動詞の原型のみ
        )

    # 
    def preprocessing(self, text):
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\r', '', text)
        text = re.sub(r'\s', '', text)
        text = text.lower()
        text = mojimoji.zen_to_han(text, kana=True)
        text = mojimoji.han_to_zen(text, digit=False, ascii=False)
        text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
        text = neologdn.normalize(text)

        return text

    # ③
    def make_sentence_list(self, sentences):
        doc = self.nlp(sentences)
        self.ginza_sents_object = doc.sents
        sentence_list = [s for s in doc.sents]

        return sentence_list

    # ④
    def make_corpus(self):
        corpus = [' '.join(self.analyzer.analyze(str(s))) + '。' for s in self.ginza_sents_object]

        return corpus
    

algorithm_dic = {"lex": LexRankSummarizer(), "tex": TextRankSummarizer(), "lsa": LsaSummarizer(),\
                  "kl": KLSummarizer(), "luhn": LuhnSummarizer(), "redu": ReductionSummarizer(),\
                  "sum": SumBasicSummarizer()}

def summarize_sentences(sentences, sentences_count=10, algorithm="lex", language="japanese"):
  corpus_maker = JapaneseCorpus()
  preprocessed_sentences = corpus_maker.preprocessing(sentences)
  preprocessed_sentences_list = corpus_maker.make_sentence_list(preprocessed_sentences)
  corpus = corpus_maker.make_corpus()
  parser = PlaintextParser.from_string(" ".join(corpus), Tokenizer(language))

  try:
    summarizer = algorithm_dic[algorithm]
  except KeyError:
    print("algorithm name:'{}'is not found.".format(algorithm))

  summarizer.stop_words = get_stop_words(language)
  sentences_count = int((len(corpus)+9)/10*3)
  summary = summarizer(document=parser.document, sentences_count=sentences_count)

  return "".join([str(preprocessed_sentences_list[corpus.index(sentence.__str__())]) for sentence in summary])


def main(text):
  #text = """要約のテスト用の文章はここに入れる"""
  algorithm = "sum"
  language = "japanese"
  sum_sentences = summarize_sentences(text, algorithm=algorithm, language=language)
  #print(sum_sentences.replace('。', '。\n'))
  #UIUX.summary.insert(sum_sentences.replace('。', '。\n'))
  return sum_sentences.replace('。', '。\n')
  

if __name__ == "__main__":
  args = sys.argv
  if(len(args) > 1):
    main(args[1])
  else:
    print("Usage: 引数で要約したい文章を入力")

