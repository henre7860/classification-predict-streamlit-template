



def clean_string(message):

    stop_words = stopwords.words('english')
    punct = string.punctuation
    stemmer1 = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    tweet = re.sub(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?',' ',message)
    tweet = re.sub('[^a-zA-Z]',' ',tweet)
    tweet = t1.lower().split()
    
    tweet=[stemmer1.stem(word) for word in tweet if (word not in stop_words) and (word not in punct)]
    tweet = [lemmatizer.lemmatize(word) for word in tweet]
    tweet=' '.join(tweet)
    
    return tweet