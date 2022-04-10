import numpy as np
import nltk
import re
from newspaper import Article
nltk.download('punkt')
#url= 'https://timesofindia.indiatimes.com/business/india-business/rbi-reduces-repo-rate-rate-by-75-basis-points-to-4-4-key-points/articleshow/74840356.cms'
url = 'https://indianexpress.com/article/business/economy/red-flags-over-pli-scheme-niti-plans-to-monitor-performance-7861991/'
article = Article(url, language="en") # en for English 
article.download() 
article.parse() 
news = article.text
myString = re.sub(r"[\n\t]*", "", news)
print(myString)
f2 = open("nogap.txt","w+")
f2.write(myString)
f2.close()


article.nlp()
print("Article Title:") 
print(article.title) #prints the title of the article
print("\n") 
print("Article Text:") 
print(article.text) #prints the entire text of the article
print("\n") 
print("Article Summary:") 
print(article.summary) #prints the summary of the article
print("\n") 
print("Article Keywords:")
print(article.keywords) #prints the keywords of the article

#write newstext file
file1=open("NewsFile.txt", "w+")
file1.write("Title:\n")
file1.write(article.title)
file1.write("\n\nArticle Text:\n")
file1.write(article.text)
file1.write("\n\n\nArticle Keywords:\n")
keywords='\n'.join(article.keywords)
file1.write(keywords)
file1.close()
