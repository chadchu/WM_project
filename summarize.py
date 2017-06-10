import sys
import numpy as np
import jieba
import jieba.analyse as ja
from scipy.spatial import distance

# split the article into sentences and cut the sentences into words
def get_words(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    text = text.replace('？', '。')
    text = text.replace('?', '。')

    sentences = text.split('。')

    stop_words = ': \n/.,?。，？：！，、的是一在有個我不了這他也就人都說而們你要之會對及和與以很種中';
    words = []
    for i in range(len(sentences)):
        s = list(jieba.cut(sentences[i]))
        for j in s:
            if j in stop_words:
                s.remove(j)
        words.append(s)

    return sentences, words

def get_vocabs(words):
    v = set()
    for i in words:
        v = v|set(i)
    return list(v)

def freq(w, s):
    return s.count(w)

def BM25(words, vocab, k_1, b):
    mat = []
    for s in words:
        mat.append( [ freq(i,s) for i in vocab ] )
    mat = np.array(mat, dtype=int)

    sentence_len = mat.sum(axis=1)
    avgl = sentence_len.mean()
    # print(avgl)
    df = mat.astype(bool).sum(axis=0)
    # print(df)
    idf = [ np.log((mat.shape[0]-i+.5)/i+.5) for i in df ]

    ret = np.zeros(mat.shape)

    for i in range(mat.shape[0]):
        l = mat[i].sum()
        for j in range(mat.shape[1]):
            ret[i, j] = idf[j] * ( mat[i, j] * (k_1+1) ) / ( mat[i, j] + k_1*(1-b+b*l/avgl) )

    return ret

# calculate cosine similarity as TextRank's weights
def get_weight(mat):

    w = np.zeros((mat.shape[0], mat.shape[0]))
    for i in range(w.shape[0]):
        for j in range(w.shape[0]):
            if i >= j:
                continue
            dot = mat[i].dot(mat[j])
            if np.any(mat[i]) and np.any(mat[i]) and dot != 0:
                cos = 1 - distance.cosine(mat[i], mat[j])
                w[i, j] = cos
                w[j, i] = cos

    return w, w.sum(axis=0)

# use TextRank to calculate the scores of each sentences
def TextRank(num, w, w_sum, d, min_diff, iteration):

    vertex = np.ones(num)

    for it in range(iteration):

        tmp = np.zeros(num)
        max_diff = 0.0

        for i in range(num):

            tmp[i] = 1 - d

            for j in range(num):

                if i  == j or w_sum[j] == 0:
                    continue

                tmp[i] += (d * w[i, j] / w_sum[j] * vertex[j])

            diff = np.absolute(tmp[i] - vertex[i])
            if diff > max_diff:
                max_diff = diff

        vertex = tmp

        # print(max_diff)
        # if max_diff <= min_diff:
            # break

    return vertex


def main():

    sentences, words = get_words(sys.argv[1])
    vocab = get_vocabs(words)
    # print(words)
    # print(vocab)
    # print(freq('勇士', words[-2]))
    # print(BM25(words, vocab, 1.2, 0.75))
    mat = BM25(words, vocab, 1.5, .75)
    weight, weight_sum = get_weight(mat)
    # print(weight, weight_sum)
    tr = TextRank(weight.shape[0], weight, weight_sum, .85, 0.5, 20)
    # print(tr)
    ranking = np.argsort(tr)[::-1]
    for i in range(3):
        print(sentences[ ranking[i] ])    

if __name__ == '__main__':
    main()
