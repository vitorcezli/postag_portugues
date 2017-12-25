#!/usr/bin/python3


def converte_palavra_lista(palavra, qc):
	"Retorna uma lista com os caracteres mais à direita da palavra"
	if len(palavra) > qc:
		return [c for c in palavra[len(palavra) - qc :]]
	else:
		return ['^'] * (qc - len(palavra)) + [c for c in palavra]