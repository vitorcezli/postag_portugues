import codecs
import math
import random


def converte_palavra_lista(palavra, qc):
	"Retorna uma lista com os caracteres mais à direita da palavra"
	if len(palavra) > qc:
		return [c for c in palavra[len(palavra) - qc :]]
	else:
		return ['^'] * (qc - len(palavra)) + [c for c in palavra]


def __separa_palavras_tags(palavras_tags):
	"Retorna uma lista contendo as palavras e as tags"
	palavras = [pt[: pt.index('_')] for pt in palavras_tags]
	tags = [pt[pt.index('_') + 1 :] for pt in palavras_tags]
	return [palavras, tags]


def __divide_subsets(valores_set, tamanho):
	if len(valores_set) <= tamanho:
		return [(v,) for v in valores_set]
	tamanho_set = len(valores_set) // tamanho
	sets_retorno = list()
	for i in range(tamanho):
		if i == tamanho - 1:
			sets_retorno.append(tuple(valores_set))
			break
		set_criado = set(random.sample(valores_set, tamanho_set))
		sets_retorno.append(tuple(set_criado))
		valores_set -= set_criado
	return sets_retorno


def __divide_sets(arquivo, tamanho_divisao):
	linhas = codecs.open(arquivo, 'r').read().split('\n')[:-1]
	tags_dict = dict()
	for linha in linhas:
		palavras, tags = __separa_palavras_tags(linha.split())
		for palavra, tag in zip(palavras, tags):
			tags_dict[tag] = tags_dict.get(tag, [])
			tags_dict[tag].append(palavra)
	sets_dict = dict()
	for key in tags_dict:
		if len(set(tags_dict[key])) > tamanho_divisao:
			sets_dict[key] = __divide_subsets(set(tags_dict[key]), tamanho_divisao)
	return sets_dict


def treinamento_validacao(arquivo, tamanho_divisao, numeracao):
	treinamento = list()
	validacao = list()
	sets_dict = __divide_sets(arquivo, tamanho_divisao)
	linhas = codecs.open(arquivo, 'r').read().split('\n')[:-1]
	for linha in linhas:
		palavras, tags = __separa_palavras_tags(linha.split())
		se_validacao = False
		for palavra, tag in zip(palavras, tags):
			if tag not in sets_dict:
				continue
			if palavra in sets_dict[tag][numeracao]:
				validacao.append(linha)
				se_validacao = True
				break
		if not se_validacao:
			treinamento.append(linha)
	return treinamento, validacao


def leia_palavras_postags(linhas):
	"Lê os dados do corpus retornando listas de palavras e classificações de 'part-of-speech'"
	return [__separa_palavras_tags(linha.split()) for linha in linhas]


def leia_postag_por_palavra(linhas):
	"Retorna todas as palavras como lista junto de sua classificação"
	pt = set([(c.split('_')[0], c.split('_')[1]) for l in linhas for c in l.split()])
	return [converte_palavra_lista(c[0], 6) + [c[1]] for c in pt]
