#download all the necessary tools and upload the input file and give the file Name in line No.237
#then run the code. it will give the output file.(for all link it take some time.)

#required Modules.
import nltk
import numpy as np
import pandas as pd
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup
nltk.download('cmudict')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import opinion_lexicon
from nltk.tokenize import sent_tokenize
nltk.download('opinion_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from textblob import TextBlob
# code for extract the content from link.
def link_content(link):
  info=requests.get(link)
  soup=BeautifulSoup(info.content,'lxml')
  conte=soup.find_all('p')
  clean_st1=str(conte)
  return clean_st1


# function for Positive score.
def pos_score(link):
  clean_st1=link_content(link)
  clean_st1=re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", clean_st1)
  clean_text=''
  for word in clean_st1.split():
     if word not in set(stopwords.words('english')):
         clean_text=clean_text+' '+word
  lemmatizer = WordNetLemmatizer()
#clean_txt= train['clean_tweet'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
  clean_txt1=''
  for word in clean_text.split():
      clean_txt1=clean_txt1+' '+lemmatizer.lemmatize(word)
  

  positive_wds=set(opinion_lexicon.positive())
  p_cnt=0
  for words in clean_txt1.split():
     if words in positive_wds:
        p_cnt=p_cnt+1
  return p_cnt
#function for Negative score score.

def neg_score(link):
  clean_st1=link_content(link)
  clean_st1=re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", clean_st1)
  clean_text=''
  for word in clean_st1.split():
     if word not in set(stopwords.words('english')):
         clean_text=clean_text+' '+word
  lemmatizer = WordNetLemmatizer()
#clean_txt= train['clean_tweet'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
  clean_txt1=''
  for word in clean_text.split():
      clean_txt1=clean_txt1+' '+lemmatizer.lemmatize(word)
  

  
  negative_wds=set(opinion_lexicon.negative())
  n_cnt=0
  for words in clean_txt1.split():
     if words in negative_wds:
        n_cnt=n_cnt+1
  return n_cnt
#function for polarity score

def pol_score(link):
   clean_st1=link_content(link)
   clean_st1=re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", clean_st1)
   clean_text=''
   for word in clean_st1.split():
      if word not in set(stopwords.words('english')):
         clean_text=clean_text+' '+word
   lemmatizer = WordNetLemmatizer()

   clean_txt1=''
   for word in clean_text.split():
      clean_txt1=clean_txt1+' '+lemmatizer.lemmatize(word)
   ans=TextBlob(clean_txt1).sentiment
   ans1=list(ans)
   return ans1[0]
  
#function for SUBJECTIVITY SCORE 
  
def sub_score(link):
   clean_st1=link_content(link)
   clean_st1=re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", clean_st1)
   clean_text=''
   for word in clean_st1.split():
      if word not in set(stopwords.words('english')):
         clean_text=clean_text+' '+word
   lemmatizer = WordNetLemmatizer()
#clean_txt= train['clean_tweet'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
   clean_txt1=''
   for word in clean_text.split():
      clean_txt1=clean_txt1+' '+lemmatizer.lemmatize(word)
   ans=TextBlob(clean_txt1).sentiment
   ans1=list(ans)
   return ans1[1]
#function for AVG SENTENCE LENGTH
def avg_sen(link):

  info=requests.get(link)
  soup=BeautifulSoup(info.content,'lxml')
  conte=soup.find_all('p')
  clean_st1=str(conte)
  
  sentence=clean_st1.split('.')
  sen_count=len(sentence)
  total_len=len(clean_st1)

  avg_sentence_len=int(total_len/sen_count)
  return avg_sentence_len
#function for PERCENTAGE OF COMPLEX WORDS
def per_complex(link):
  clean_st1=link_content(link)
  clean_st1=re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", clean_st1)
  clean_text=''
  for word in clean_st1.split():
     if word not in set(stopwords.words('english')):
         clean_text=clean_text+' '+word
  lemmatizer = WordNetLemmatizer()
#clean_txt= train['clean_tweet'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
  clean_txt1=''
  for word in clean_text.split():
     clean_txt1=clean_txt1+' '+lemmatizer.lemmatize(word)


  cnt=1
  for word in clean_text.split():
     if len(word)>7:
        cnt=cnt+1
  #total_words=5
  total_words=len(clean_text.split())
  if total_words<1:
    total_words=cnt*0.12
  percentage=int((cnt/total_words)*100)
  return percentage

#function for FOG INDEX
# average sentence length + percentage.

#function for AVG NUMBER OF WORDS PER SENTENCE

def avg_word(link):
  info=requests.get(link)
  soup=BeautifulSoup(info.content,'lxml')
  conte=soup.find_all('p')
  clean_st1=str(conte)
  sentence=clean_st1.split('.')
  sen_count=len(sentence)
  total_len=len(clean_st1.split(' '))
  avg_sentence_len=int(total_len/sen_count)
  return avg_sentence_len

#function for COMPLEX WORD COUNT


def complex_word(link):
  clean_st1=link_content(link)
  clean_st1=re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", clean_st1)
  clean_text=''
  for word in clean_st1.split():
     if word not in set(stopwords.words('english')):
         clean_text=clean_text+' '+word
  lemmatizer = WordNetLemmatizer()
  clean_txt1=''
  for word in clean_text.split():
     clean_txt1=clean_txt1+' '+lemmatizer.lemmatize(word)


  cnt=1
  for word in clean_text.split():
     if len(word)>7:
        cnt=cnt+1
  return cnt

#function for WORD COUNT

def word_count(link):
  clean_st1=link_content(link)
  clean_st1=re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", clean_st1)
  clean_text=''
  for word in clean_st1.split():
     if word not in set(stopwords.words('english')):
         clean_text=clean_text+' '+word
  lemmatizer = WordNetLemmatizer()
  clean_txt1=''
  for word in clean_text.split():
     clean_txt1=clean_txt1+' '+lemmatizer.lemmatize(word)
  return len(clean_txt1.split())

#function for PERSONAL PRONOUNS
def find_personal_pronouns(link):
   clean_st1=link_content(link)
   clean_st1=re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", clean_st1)
   clean_text=''
   for word in clean_st1.split():
     if word not in set(stopwords.words('english')):
         clean_text=clean_text+' '+word
   lemmatizer = WordNetLemmatizer()
   clean_txt1=''
   for word in clean_text.split():
     clean_txt1=clean_txt1+' '+lemmatizer.lemmatize(word)
   words = nltk.word_tokenize(clean_txt1)
   pos_tags = nltk.pos_tag(words)
   personal_pronouns = [word for word, pos in pos_tags if pos in ['PRP', 'PRP$', 'WP', 'WP$']]
   return personal_pronouns

#function for AVG WORD LENGTH
def word_count(link):
  clean_st1=link_content(link)
  clean_st1=re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", clean_st1)
  clean_text=''
  for word in clean_st1.split():
     if word not in set(stopwords.words('english')):
         clean_text=clean_text+' '+word
  lemmatizer = WordNetLemmatizer()
  total_len=len(clean_text)
  word_len=len(clean_text.split(' '))
  avg_word_len=int(total_len/word_len)
  return avg_word_len

# *************upload the file and give the path *****************

df=pd.read_excel('input_file_Name.xlsx')
#************* HERE I assume that one column of name URL.
df1=df['URL']
# every function is called and it add the value .
df['positive score']=df1.apply(pos_score)
df['Negative score']=df1.apply(neg_score)
df['polarity score']=df1.apply(pol_score)
df['SUBJECTIVITY SCORE']=df1.apply(sub_score)
df['AVG SENTENCE LENGTH']=df1.apply(avg_sen)
df['PERCENTAGE OF COMPLEX WORDS']=df1.apply(per_complex)
df['FOG INDEX']=df['AVG SENTENCE LENGTH']+df['PERCENTAGE OF COMPLEX WORDS']
df['AVG NUMBER OF WORDS PER SENTENCE']=df1.apply(avg_word)
df['COMPLEX WORD COUNT']=df1.apply(complex_word)
df['WORD COUNT']=df1.apply(complex_word)
df['PERSONAL PRONOUNS']=df1.apply(find_personal_pronouns)
df['AVG WORD LENGTH']=df1.apply(word_count)
#After complete the process this is the code for download the output file.
df.to_excel('Output Data Structure.xlsx', index=False)

  
