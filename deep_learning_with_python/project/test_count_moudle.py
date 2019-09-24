import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

imdb_dir = 'D:\BaiduNetdiskDownload\\aclImdb\\aclImdb'
test_dir = os.path.join(imdb_dir,'test')
labels = []
texts = []

maxlen = 100
max_words = 10000

for label_type in ['neg','pos']:
    dir_name = os.path.join(test_dir,label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name,fname),encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
tokenizer = Tokenizer(num_words=max_words)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences,maxlen = maxlen)
y_test = np.asarray(labels)

model = Sequential()
model.load_weights('pre_trained_glove_model.h5')
print(model.evaluate(x_test, y_test))