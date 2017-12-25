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
		tags = [self.observacao.get((tag, palavra), 0) / self.unigram[tag] \
			for tag in self.tags]
		return tags


	def __pega_transicao(self, tag1, tag2):
		"Retorna a probabilidade de transição entre tags"
		return self.bigram.get((tag1, tag2), 0) / self.unigram.get(tag1, 0)	


	def __pega_maior_indice_valor(self, matriz, estado, tag):
		"Retorna a maior probabilidade e o índice em um estado no algoritmo de Viterbi"
		valores = [matriz[k][estado] * self.__pega_transicao(self.tags[k], tag) \
			for k in range(len(self.tags))]
		return [max(valores), valores.index(max(valores))]


	def __diminui_risco_underflow(self, matriz, estado):
		"Reduz o risco de overflow nos estados multiplicando a probabilidade"
		if estado % 5 == 0:
			for linha in range(len(matriz)):
				matriz[linha][estado] *= 100000


	def __pega_pos(self, ultima_coluna, ponteiros):
		"Retorna classificações 'part-of-speech' utilizando os ponteiros"
		ponteiros_inversos = [ultima_coluna.index(max(ultima_coluna))]
		for i in range(len(ponteiros[0]) - 1, -1, -1):
			ponteiros_inversos.append(ponteiros[-1][i])
		return list(reversed([self.tags[indice] for indice in ponteiros_inversos]))[: -1]


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
				[estados[i][j], ponteiros[i][j - 1]] = \
					self.__pega_maior_indice_valor(estados, j - 1, self.tags[i])
				estados[i][j] *= observacoes[i]
			self.__diminui_risco_underflow(estados, j)
		# calcula os estados finais
		for i in range(len(self.tags)):
			[estados[i][len(frase)], ponteiros[i][len(frase) - 1]] = \
				self.__pega_maior_indice_valor(estados, len(frase) - 1, self.tags[i])
		# usa os ponteiros para definir part-of-speech
		return self.__pega_pos([linha[len(linha) - 1] for linha in estados], ponteiros)
		


anotador = classificador_postag()
texto = 'O grande assunto da semana em Nova York é a edição da revista \" New Yorker \" que está nas bancas .'
print(texto.split())
print(anotador.classifica(texto.split()))