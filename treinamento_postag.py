#!/usr/bin/python3
from leitor_postag import leia_palavras_postags
from leitor_postag import leia_postag_por_palavra
from sklearn.svm import SVC
import copy
import numpy as np
import pickle


# lê as palavras e as classificações do arquivo
lista_palavras_tags = leia_palavras_postags('macmorpho-train.txt')
# variáveis utilizadas para armazenar dados que serão utilizados para treinar
# o classificador por palavras e executar o algoritmo Viterbi
observacoes = {}
bigrams = {}
unigrams = {}
# armazena os dados
for linha in lista_palavras_tags:
	palavras = linha[0]
	tags = linha[1]
	# armazena as observações
	for i in range(len(tags)):
		observacoes[tags[i], palavras[i]] = observacoes.get((tags[i], palavras[i]), 0) + 1
	# armazena unigrams
	unigrams['<s>'] = unigrams.get('<s>', 0) + 1
	for i in range(len(tags)):
		unigrams[tags[i]] = unigrams.get(tags[i], 0) + 1
	# armazena bigrams
	bigrams['<s>', tags[0]] = bigrams.get(('<s>', tags[0]), 0) + 1
	for i in range(1, len(tags)):
		bigrams[tags[i - 1], tags[i]] = bigrams.get((tags[i - 1], tags[i]), 0) + 1
	bigrams[tags[len(tags) - 1], '</s>'] = bigrams.get((tags[len(tags) - 1], '</s>'), 0) + 1
# lê as palavras do corpus em forma de lista e com a classificação de
# postag na última coluna
palavras_tags = leia_postag_por_palavra('macmorpho-train.txt')
# a variável tags é utilizada para indexar as classificações
tags = list(set(linha[-1] for linha in palavras_tags))
# as árvores de decisão de Random Forest necessitam de valores numéricos
# nos atributos e nas classificações
data_x = [[ord(c) for c in linha[: -1]] for linha in palavras_tags]
data_y = [tags.index(linha[-1]) for linha in palavras_tags]
# treina o classificador de RandomForest
classificador = SVC()
classificador.fit(data_x, data_y)
# salva os dados do classificador e das informações para o algoritmo Viterbi
pickle.dump(classificador, open('classificador.sav', 'wb'))
pickle.dump(observacoes, open('observacao.sav', 'wb'))
pickle.dump(unigrams, open('unigram.sav', 'wb'))
pickle.dump(bigrams, open('bigram.sav', 'wb'))
pickle.dump(tags, open('tags.sav', 'wb'))
