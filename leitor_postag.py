#!/usr/bin/python3
from conversor_postag import converte_palavra_lista
import codecs


def __separa_palavras_tags(palavras_tags):
	"Retorna uma lista contendo as palavras e as tags"
	palavras = [pt[: pt.index('_')] for pt in palavras_tags]
	tags = [pt[pt.index('_') + 1 :] for pt in palavras_tags]
	return [palavras, tags]


def leia_palavras_postags(arquivo):
	"Lê os dados do corpus retornando listas de palavras e classificações de 'part-of-speech'"
	linhas = codecs.open(arquivo, 'r').read().split('\n')[:-1]
	return [__separa_palavras_tags(linha.split()) for linha in linhas]


def leia_postag_por_palavra(arquivo):
	"Retorna todas as palavras como lista junto de sua classificação"
	linhas = codecs.open(arquivo, 'r').read().split('\n')[:-1]
	pt = set([(c.split('_')[0], c.split('_')[1]) for l in linhas for c in l.split()])
	return [converte_palavra_lista(c[0], 6) + [c[1]] for c in pt]
