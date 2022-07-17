"""
This file contains APIs and pre-defined variables used to preprocess data and other helper functions used in data visualization.
"""

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
import os
from __future__ import print_function
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
import pandas as pd
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis.gensim_models
import pickle
import pyLDAvis
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'coronavirus','covid'])
import gensim.corpora as corpora
from pprint import pprint
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

# Some Gibberish characters to be removed
puncts = ['☹', 'Ź', 'Ż', 'ἰ', 'ή', 'Š', '＞', 'ξ','ฉ', 'ั', 'น', 'จ', 'ะ', 'ท', 'ำ', 'ใ', 'ห', '้', \
'ด', 'ี', '่', 'ส', 'ุ', 'Π', 'प', 'ऊ', 'Ö', 'خ', 'ب', 'ஜ', 'ோ', 'ட', '「', 'ẽ', '½', '△', 'É', \
'ķ', 'ï', '¿', 'ł', '북', '한', '¼', '∆', '≥', '⇒', '¬', '∨', 'č', 'š', '∫', 'ḥ', 'ā', 'ī', 'Ñ', \
'à', '▾', 'Ω', '＾', 'ý', 'µ', '?', '!', '.', ',', '"', '#', '$', '%', '\\', "'", '(', ')', '*', \
'+', '-', '/', ':', ';', '<', '=', '>', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', '“', '”', \
'’', 'é', 'á', '′', '…', 'ɾ', '̃', 'ɖ', 'ö', '–', '‘', 'ऋ', 'ॠ', 'ऌ', 'ॡ', 'ò', 'è', 'ù', 'â', 'ğ', \
'म', 'ि', 'ल', 'ग', 'ई', 'क', 'े', 'ज', 'ो', 'ठ', 'ं', 'ड', 'Ž', 'ž', 'ó', '®', 'ê', 'ạ', 'ệ', '°', \
'ص', 'و', 'ر', 'ü', '²', '₹', 'ú', '√', 'α', '→', 'ū', '—', '£', 'ä', '️', 'ø', '´', '×', 'í', 'ō', \
'π','÷', 'ʿ', '€', 'ñ', 'ç', 'へ', 'の', 'と', 'も', '↑', '∞', 'ʻ', '℅''ι', '•', 'ì', '−', 'л', 'я', 'д', \
    'ل', 'ك', 'م', 'ق', 'ا', '∈', '∩', '⊆', 'ã', 'अ', 'न', 'ु', 'स', '्', 'व', 'ा', 'र', 'त', '§', '℃', \
    'θ', '±', '≤', 'उ', 'द', 'य', 'ब', 'ट', '͡', '͜', 'ʖ', '⁴', '™', 'ć', 'ô', 'с', 'п', 'и', 'б', 'о', 'г', \
    '≠', '∂', 'आ', 'ह', 'भ', 'ी', '³', 'च', '...', '⌚', '⟨', '⟩', '∖', '˂', 'ⁿ', '⅔', 'న', 'ీ', 'క', 'ె', \
    'ం', 'ద', 'ు', 'ా', 'గ', 'ర', 'ి', 'చ', 'র', 'ড়', 'ঢ়', 'સ', 'ં', 'ઘ', 'ર', 'ા', 'જ', '્', 'ય', 'ε', \
    'ν', 'τ', 'σ', 'ş', 'ś', 'س', 'ت', 'ط', 'ي', 'ع', 'ة', 'د', 'Å', '☺', 'ℇ', '❤', '♨', '✌', 'ﬁ', 'て', \
    '„', 'Ā', 'ត', 'ើ', 'ប', 'ង', '្', 'អ', 'ូ', 'ន', 'ម', 'ា', 'ធ', 'យ', 'វ', 'ី', 'ខ', 'ល', 'ះ', 'ដ', \
    'រ', 'ក', 'ឃ', 'ញ', 'ឯ', 'ស', 'ំ', 'ព', 'ិ', 'ៃ', 'ទ', 'គ', '¢', 'つ', 'や', 'ค', 'ณ', 'ก', 'ล', 'ง', \
    'อ', 'ไ', 'ร', 'į', 'ی', 'ю', 'ʌ', 'ʊ', 'י', 'ה', 'ו', 'ד', 'ת', 'ᠠ', 'ᡳ', 'ᠰ', 'ᠨ', 'ᡤ', 'ᡠ', 'ᡵ', 'ṭ', \
    'ế', 'ध', 'ड़', 'ß', '¸', 'ч',  'ễ', 'ộ', 'फ', 'μ', '⧼', '⧽', 'ম', 'হ', 'া', 'ব', 'ি', 'শ', '্', 'প', \
    'ত', 'ন', 'য়', 'স', 'চ', 'ছ', 'ে', 'ষ', 'য', '়', 'ট', 'উ', 'থ', 'ক', 'ῥ', 'ζ', 'ὤ', 'Ü', 'Δ', '내', \
    '제', 'ʃ', 'ɸ', 'ợ', 'ĺ', 'º', 'ष', '♭', '़', '✅', '✓', 'ě', '∘', '¨', '″', 'İ', '⃗', '̂', 'æ', 'ɔ', '∑', \
    '¾', 'Я', 'х', 'О', 'з', 'ف', 'ن', 'ḵ', 'Č', 'П', 'ь', 'В', 'Φ', 'ỵ', 'ɦ', 'ʏ', 'ɨ', 'ɛ', 'ʀ', 'ċ', 'օ', \
    'ʍ', 'ռ', 'ք', 'ʋ', '兰', 'ϵ', 'δ', 'Ľ', 'ɒ', 'î', 'Ἀ', 'χ', 'ῆ', 'ύ', 'ኤ', 'ል', 'ሮ', 'ኢ', 'የ', 'ኝ', 'ን', \
    'አ', 'ሁ', '≅', 'ϕ', '‑', 'ả', '￼', 'ֿ', 'か', 'く', 'れ', 'ő', '－', 'ș', 'ן', 'Γ', '∪', 'φ', 'ψ', '⊨', 'β', \
    '∠', 'Ó', '«', '»', 'Í', 'க', 'வ', 'ா', 'ம', '≈', '⁰', '⁷', 'ấ', 'ũ', '눈', '치', 'ụ', 'å', '،', '＝', \
    '（', '）', 'ə', 'ਨ', 'ਾ', 'ਮ', 'ੁ', '︠', '︡', 'ɑ', 'ː', 'λ', '∧', '∀', 'Ō', 'ㅜ', 'Ο', 'ς', 'ο', 'η', 'Σ', 'ण']

odd_chars = [ '大','能', '化', '生', '水', '谷', '精', '微', 'ル', 'ー', 'ジ', 'ュ', '支', '那', '¹', 'マ', \
'リ', '仲', '直', 'り', 'し', 'た', '主', '席', '血', '⅓', '漢', '髪', '金', '茶', '訓', '読', '黒', 'ř', \
'あ', 'わ', 'る', '胡', '南', '수', '능', '广', '电', '总', 'ί', '서', '로', '가', '를', '행', '복', '하', '게', \
'기', '乡', '故', '爾', '汝', '言', '得', '理', '让', '骂', '野', '比', 'び', '太', '後', '宮', '甄', '嬛', '傳', \
'做', '莫', '你', '酱', '紫', '甲', '骨', '陳', '宗', '陈', '什', '么', '说', '伊', '藤', '長', 'ﷺ', '僕', 'だ', \
'け', 'が', '街', '◦', '火', '团', '表',  '看', '他', '顺', '眼', '中', '華', '民', '國', '許', '自', '東', '儿', \
'臣', '惶', '恐', 'っ', '木', 'ホ', 'ج', '教', '官', '국', '고', '등', '학', '교', '는', '몇', '시', '간', '업', \
'니', '本', '語', '上', '手', 'で', 'ね', '台', '湾', '最', '美', '风', '景', 'Î', '≡', '皎', '滢', '杨', '∛', \
'簡', '訊', '短', '送', '發', 'お', '早', 'う', '朝', 'ش', 'ه', '饭', '乱', '吃', '话', '讲', '男', '女', '授', \
'受', '亲', '好', '心', '没', '报', '攻', '克', '禮', '儀', '統', '已', '經', '失', '存', '٨', '八', '‛', '字', \
'：', '别', '高', '兴', '还', '几', '个', '条', '件', '呢', '觀', '《', '》', '記', '宋', '楚', '瑜', '孫', '瀛', \
'枚', '无', '挑', '剔', '聖', '部', '頭', '合', '約', 'ρ', '油', '腻', '邋', '遢', 'ٌ', 'Ä', '射', '籍', '贯', '老', \
'常', '谈', '族', '伟', '复', '平', '天', '下', '悠', '堵', '阻', '愛', '过', '会', '俄', '罗', '斯', '茹', '西', \
'亚', '싱', '관', '없', '어', '나', '이', '키', '夢', '彩', '蛋', '鰹', '節', '狐', '狸', '鳳', '凰', '露', '王', \
'晓', '菲', '恋', 'に', '落', 'ち', 'ら', 'よ', '悲', '反', '清', '復', '明', '肉', '希', '望', '沒', '公', '病', \
'配', '信', '開', '始', '日', '商', '品', '発', '売', '分', '子', '创', '意', '梦', '工', '坊', 'ک', 'پ', 'ڤ', '蘭', \
'花', '羡', '慕', '和', '嫉', '妒', '是', '样', 'ご', 'め', 'な', 'さ', 'い', 'す', 'み', 'ま', 'せ', 'ん', '音', \
'红', '宝', '书', '封', '柏', '荣', '江', '青', '鸡', '汤', '文', '粵', '拼', '寧', '可', '錯', '殺', '千', '絕', \
'放', '過', '」', '之', '勢', '请', '国', '知', '识', '产', '权', '局', '標', '點', '符', '號', '新', '年', '快', \
'乐', '学', '业', '进', '步', '身', '体', '健', '康', '们', '读', '我', '的', '翻', '译', '篇', '章', '欢', '迎', \
'入', '坑', '有', '毒', '黎', '氏', '玉', '英', '啧', '您', '这', '口', '味', '奇', '特', '也', '就', '罢', '了', \
'非', '要', '以', '此', '为', '依', '据', '对', '人', '家', '批', '判', '一', '番', '不', '地', '道', '啊', '谢', \
'六', '佬']

# Dictionary to help with converting informal expressions into formal ones
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "can't've": "cannot have", 
"'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
"didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", 
"hasn't": "has not", "haven't": "have not",  "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
"he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
"how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
"I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
"i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 
"it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
"might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", 
"mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
"o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
"sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 
"she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", 
"shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is",
"that'd": "that would", "that'd've": "that would have","that's": "that is", "there'd": "there would", 
"there'd've": "there would have","there's": "there is", "here's": "here is","they'd": "they would", 
"they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", 
"they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", 
"we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", 
"what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", 
"what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", 
"where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", 
"why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", 
"would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 
"y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
"you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", 
"you're": "you are", "you've": "you have" } 

# Function to convert contractions into their full form based on a previously defined dictionary
def clean_contractions(text:str):
    """
    Parameters
    ----------
    text : the input STRING to clean up
    """
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
    return text

# Function to clean up odd characters in a previously defined list
def odd_add_space(x:str):
    """
    Parameters
    ----------
    x : the input STRING to clean up
    """
    x = str(x)
    for odd in odd_chars:
        x = x.replace(odd, f' {odd} ')
    return x 

# Function to cleans out all numbers from text
def clean_numbers(x:str):
    """
    Parameters
    ----------
    x : the input STRING to clean up
    """
    x = re.sub('[0-9]{5,}', ' ##### ', x)
    x = re.sub('[0-9]{4}', ' #### ', x)
    x = re.sub('[0-9]{3}', ' ### ', x)
    x = re.sub('[0-9]{2}', ' ## ', x)
    return x

# Function to remove all punctuations in a previously defined list
def punct_add_space(x):
    """
    Parameters
    ----------
    x : the input STRING to clean up
    """
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x  

# Function to remove punctuation from the question text
def remove_punctuation(text:str):
    """
    Parameters
    ----------
    text : the input STRING to remove punctuation from
    """
    return re.sub(r'[^\w\s]', '', text)

# Function to remove stopwords from the question text
def remove_stopwords(words:list):
    """
    Parameters
    ----------
    words : the input LIST to remove words from
    """
    stop_words = set(stopwords.words("english"))
    return [word for word in words if word not in stop_words]

# Function to lemmatize text (return base form of words)
def lemmatize_text(words:list):
    """
    Parameters
    ----------
    words : the input LIST to lemmatize
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

# Function to stem the text (return stem of words)
def stem_text(words:list):
    """
    Parameters
    ----------
    words : the input LIST to stem
    """
    ps = PorterStemmer()
    return [ps.stem(word) for word in words]

# Final function, which uses all the helper functions defined above, used to preprocess text
def preprocess_text(text):

    text = clean_contractions(text)
    text = odd_add_space(text)
    text = punct_add_space(text)
    text = clean_numbers(text)
    text = remove_punctuation(text).split()

    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = " ".join(stem_text(text))

    return text


"""function to generate unigram:
  - Input : 
    - Corpus: A list of words in a corpus
  - Output : 
    - words_freq[:n] : A list of the unique words and their frequency in database"""

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

"""function to generate additional stopwords:
  - Input : 
    - common words: A list of commonly occuring stopwords 
    - remove_list : Existing list of stopwords 
  - Output : 
    - remove_list : Updated list of stopwords
    - words_freq[:n] : A list of the unique words and their frequency in database"""

def generate_stopwords(common_words,remove_list):
    newStopWords = []
    for word, freq in common_words:
        newStopWords.append(word)
    newStopWords.remove('china')
    newStopWords.remove('indian')
    newStopWords.remove('vaccine')

    newStopWords.remove('trump')
    newStopWords.remove('italy')
    newStopWords.remove('uk')
    newStopWords.remove('police')

    newStopWords.remove('patients')
    newStopWords.remove('000')
    newStopWords.remove('oil')
    newStopWords.remove('deaths')

    newStopWords.remove('home')
    newStopWords.remove('facebook')
    newStopWords.remove('spain')
    remove_list.extend(newStopWords)
    return remove_list

"""function to generate bigrams:
  - Input : 
    - Corpus: A list of words in a corpus
  - Output : 
    - words_freq[:n] : A list of the unique bigrams and their frequency in database"""

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words=remove_list).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


"""function to generate trigrams:
  - Input : 
    - Corpus: A list of words in a corpus
  - Output : 
    - words_freq[:n] : A list of the unique bigrams and their frequency in database"""

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]




"""
    Function to convert a document into a list of tokens. This lowercases, tokenizes, de-accents (optional). – the output are final tokens = unicode strings, that won’t be processed any further.
        - Input  : text 
        - Output : list of processed tokens 
    
"""
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


"""
        Function to create a BoW corpus from a simple list of documents and from text file
        - Input  : text 
        - Output : BoW corpus 
"""
def generate_Data_words(data_words):
    id2word = corpora.Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    return corpus



"""
        Function to remove stopwords from text
        - Input  : Text
        - Output : List of words in text without stopwords
"""
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

"""
-Input: 
    - final_clean : Input dataframe containing a reliability and content column
- Output :
    - Grouped Dataframe : Dataframe grouped by reliability score 
"""
def group_by_reliability(final_clean):
    group = final_clean.groupby('reliability')
    df2 = group.apply(lambda x: x['clean_content'].unique())
    grouped_df = pd.DataFrame(df2)
    grouped_df = grouped_df.reset_index().rename(columns={0: 'content'})
    return grouped_df

"""
        Function to create a BoW corpus for an input into the LDA model from a simple list of documents and from text file
        - Input  : text 
        - Output : BoW corpus 
"""

def generate_lda_corpus(inputval):
    df = pd.read_csv("final_clean.csv")
    group = df.groupby('reliability')
    df2 = group.apply(lambda x: x['clean_content'].unique())
    df3 = pd.DataFrame(df2)
    df3 = df3.reset_index().rename(columns={0: 'clean_content'})
    data = df3.clean_content[inputval]
    data_words = list(sent_to_words(data))
    corpus = generate_Data_words(data_words)
    return corpus

"""function to generate intertopic distance plot:
  - Input : 
    - Corpus: A list of words in a corpus
    - num_topics : Number of topics to be generated 
    - lda_model : lda_model generated 
    - id2word : texts in topic model
  - Output : 
    - words_freq[:n] : A list of the unique bigrams and their frequency in database"""

def generate_lda_graph(num_topics,lda_model,corpus,id2word):
  pyLDAvis.enable_notebook()
  LDAvis_data_filepath = os.path.join('ldavis_prepared_'+str(num_topics))
  if 1 == 1:
      LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
      with open(LDAvis_data_filepath, 'wb') as f:
          pickle.dump(LDAvis_prepared, f)
  # load the pre-prepared pyLDAvis data from disk
  with open(LDAvis_data_filepath, 'rb') as f:
      LDAvis_prepared = pickle.load(f)
  pyLDAvis.save_html(LDAvis_prepared, 'ldavis_prepared_'+ str(num_topics) +'.html')
  LDAvis_prepared
  return LDAvis_prepared



