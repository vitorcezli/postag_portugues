import copy
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

import leitor_postag


def treina_hmm(treinamento):
	# lê as palavras e as classificações do arquivo
	lista_palavras_tags = leitor_postag.leia_palavras_postags(treinamento)
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
	palavras_tags = leitor_postag.leia_postag_por_palavra(treinamento)
	# a variável tags é utilizada para indexar as classificações
	tags = list(set(linha[-1] for linha in palavras_tags))
	# as árvores de decisão de Random Forest necessitam de valores numéricos
	# nos atributos e nas classificações
	data_x = [[ord(c) for c in linha[: -1]] for linha in palavras_tags]
	data_y = [tags.index(linha[-1]) for linha in palavras_tags]
	# treina o classificador de RandomForest
	classificador = RandomForestClassifier()
	classificador.fit(data_x, data_y)
	# salva os dados do classificador e das informações para o algoritmo Viterbi
	return [classificador, observacoes, unigrams, bigrams, tags]


class classificador_postag:

	def __init__(self, classificador, observacoes, unigrams, bigrams, tags):
		"Inicializa as variáveis que serão utilizadas em 'part-of-speech tagging'"
		self.unigram = unigrams
		self.bigram = bigrams
		self.observacao = observacoes
		self.tags = tags
		self.classificador = classificador
		self.tags_probabilidades = self.__tags_probabilidades()

	def __tags_probabilidades(self):
		"Retorna a probabilidade de ocorrência de cada tag"
		total = sum([valor for tag, valor in self.unigram.items() if tag != '<s>'])
		return [self.unigram[self.tags[i]] / total for i in range(len(self.tags))]

	def __e_numero(self, palavra):
		"Retorna se a palavra é um número"
		try:
			_ = float(palavra)
			return True
		except ValueError:
			pass
		return False

	def __define_probabilidade_tag(self, tag):
		"Retorna uma lista com a probabilidade de uma tag definida em 1"
		return [1 if t == tag else 0 for t in self.tags]

	def __pega_observacao(self, palavra):
		"Retorna a probabilidade de observar a palavra para cada tag"
		tags = [self.observacao.get((tag, palavra), 0) / self.unigram[tag] \
			for tag in self.tags]
		# palavra já é reconhecida
		if any(tag > 0 for tag in tags):
			return tags
		# palavra é um número
		elif self.__e_numero(palavra):
			return self.__define_probabilidade_tag('NUM')
		# testa se a mesma palavra com a primeira letra minúscula é reconhecida
		elif palavra[0].isupper():
			palavra = palavra[0].lower() + palavra[1 :]
			observacao_lower = self.__pega_observacao(palavra)
			if any(tag > 0 for tag in observacao_lower):
				return observacao_lower
		# calcula o likelihood usando o classificador que foi treinado
		posteriores = self.classificador.predict_proba([[ord(c) \
			for c in leitor_postag.converte_palavra_lista(palavra, 6)]])[0]
		return [posteriores[i] / self.tags_probabilidades[i] \
			for i in range(len(posteriores))]

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
		maximo = max([matriz[i][estado] for i in range(len(matriz))])
		if maximo != 0:
			for linha in range(len(matriz)):
				matriz[linha][estado] /= maximo

	def __pega_pos(self, ultima_coluna, ponteiros):
		"Retorna classificações 'part-of-speech' utilizando os ponteiros"
		ponteiros_inversos = [ultima_coluna.index(max(ultima_coluna))]
		for i in range(len(ponteiros[0]) - 1, -1, -1):
			ponteiros_inversos.append(ponteiros[ponteiros_inversos[-1]][i])
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


def heatmap():
	classificador = classificador_postag(0,0,0,0,0)
	tag_set = classificador.tags
	heatmap = np.zeros((len(tag_set), len(tag_set)))
	frases_tags = leitor_postag.leia_palavras_postags('macmorpho-test.txt')
	for frases, tags in frases_tags:
		resultados = classificador.classifica(frases)
		for r1, r2 in zip(tags, resultados):
			linha = tag_set.index(r1)
			coluna = tag_set.index(r2)
			heatmap[linha, coluna] += 1
	heatmap /= heatmap.sum(axis=1, keepdims=True)
	heatmap *= 100
	sns.heatmap(heatmap, xticklabels=tag_set, yticklabels=tag_set, annot=True, fmt='.1f')
	plt.show()


if __name__ == '__main__':
	for i in range(30):
		treinamento, validacao, dicts_tag = leitor_postag.treinamento_validacao('data.txt', 30, i)
		classificador, observacoes, unigrams, bigrams, tags = treina_hmm(treinamento)
		classificador = classificador_postag(classificador, observacoes, unigrams, bigrams, tags)
		frases_tags = leitor_postag.leia_palavras_postags(validacao)
		acertos = 0
		total = 0
		for index, ft in enumerate(frases_tags):
			frase, tags = ft
			resultados = classificador.classifica(frase)
			for j in range(len(resultados)):
				if tags[j] in dicts_tag and frase[j] in dicts_tag[tags[j]]:
					if resultados[j] == tags[j]:
						acertos += 1
					total += 1
			print(index, len(frases_tags))
		print(i, acertos / total)
		break
