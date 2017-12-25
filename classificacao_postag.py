#!/usr/bin/python3
from __future__ import division
import pickle
import math


# verificar features utilizando dimensão R
# treinar um classificador para as palavras, que será utilizado nas que não são reconhecidas
# dividir o resultado do classificador pela probabilidade da tag no algoritmo Viterbi
# testar o corpus de teste
class classificador_postag:


	def __init__(self):
		"Inicializa as variáveis que serão utilizadas em 'part-of-speech tagging'"
		self.unigram = pickle.load(open('unigram.sav', 'rb'))
		self.bigram = pickle.load(open('bigram.sav', 'rb'))
		self.observacao = pickle.load(open('observacao.sav', 'rb'))
		self.tags = pickle.load(open('tags.sav', 'rb'))


	def __pega_observacao(self, palavra):
		"Retorna a probabilidade de observar a palavra para cada tag"
		tags = [self.observacao.get((tag, palavra), 0) / self.unigram[tag] for tag in self.tags]
		return tags


	def __pega_transicao(self, tag1, tag2):
		"Retorna a probabilidade de transição entre tags"
		return self.bigram.get((tag1, tag2), 0) / self.unigram.get(tag1, 0)	


	def classifica(self, frase):
		"Retorna 'part-of-speech tagging' da frase passada para esta função"
		# inicializa as variáveis do algoritmo Viterbi
		estados = [[-math.inf for j in range(len(frase) + 1)] for i in range(len(self.tags))]
		ponteiros = [[-1 for j in range(len(frase))] for i in range(len(self.tags))]
		# calcula os estados iniciais
		observacoes = self.__pega_observacao(frase[0])
		for i in range(len(self.tags)):
			estados[i][0] = self.__pega_transicao('<s>', self.tags[i]) * observacoes[i]
		# calcula os estados intermediários
		for j in range(1, len(frase)):
			observacoes = self.__pega_observacao(frase[j])
			for i in range(len(self.tags)):
				for k in range(len(self.tags)):
					valor = estados[k][j - 1] * \
						self.__pega_transicao(self.tags[k], self.tags[i])
					if valor > estados[i][j]:
						estados[i][j] = valor
						ponteiros[i][j - 1] = k
				estados[i][j] *= observacoes[i]
				# esta operação é utilizada para evitar underflow
				if j % 5 == 0:
					estados[i][j] *= 100000
		# calcula os estados finais
		for i in range(len(self.tags)):
			for k in range(len(self.tags)):
				valor = estados[k][len(frase) - 1] * \
					self.__pega_transicao(self.tags[k], self.tags[i])
				if valor > estados[i][j]:
					estados[i][len(frase)] = valor
					ponteiros[i][len(frase) - 1] = k
		# usa os ponteiros para definir part-of-speech
		ultima_coluna = [linha[len(linha) - 1] for linha in estados]
		ponteiros_inversos = [ultima_coluna.index(max(ultima_coluna))]
		indice = len(ponteiros[0]) - 1
		while indice >= 0:
			ultimo_valor = ponteiros_inversos[len(ponteiros_inversos) - 1]
			ponteiros_inversos.append(ponteiros[ultimo_valor][indice])
			indice -= 1
		# define part-of-speech
		tags = []
		for i in range(len(ponteiros_inversos) - 1, 0, -1):
			tags.append(self.tags[ponteiros_inversos[i]])
		return tags


anotador = classificador_postag()
texto = 'O grande assunto da semana em Nova York é a edição da revista \" New Yorker \" que está nas bancas .'
print(texto.split())
print(anotador.classifica(texto.split()))