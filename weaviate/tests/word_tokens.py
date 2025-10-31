# import nltk
# nltk.download('all')
from nltk.tokenize import sent_tokenize, word_tokenize 
text = "Encontra contratos que ainda não passaram pela etapa de Aprovação Financeira."
  
print(sent_tokenize(text)) 
print("word tokenization")
tokenized = word_tokenize(text)
print(tokenized) 
token = 0
# run througth tokenized words to find etapa
for i , n in enumerate(tokenized):
    if n == "etapa":
        print("found", tokenized[i]," in position", i)
        token = i

# run througth words close to etapa until the end of the sentence or beginning
for i in range(1, 5):
    print("search collection etapa for object with name", tokenized[token+i])
    if tokenized[token+i] == "Aprovação":
        print("collections found with segmented name ", tokenized[token+i], ", search next word to complete the object name")
        print("search collection etapa for object with name", tokenized[token+i] + " " + tokenized[token+i+1])
        print("Perfect match found")
        print("From this object search linked objects")
        break
    print("search collection etapa for object with name", tokenized[token-i])
    if tokenized[token-i] == "Aprovação":
        print("collections found with segmented name ", tokenized[token+i], ", search next word to complete the object name")
        print("search collection etapa for object with name", tokenized[token+i] + " " + tokenized[token+i+1])
        print("Perfect match found")
        print("From this object search linked objects")
        break

