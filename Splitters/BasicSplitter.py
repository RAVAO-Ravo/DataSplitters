#!/bin/python3
#-*- coding:utf-8 -*-

# Importation des modules
import pandas as pd
from typing import Union, List, Tuple

def calculate_test_samples(n_samples: int, test_size: Union[float, int]) -> int:
	"""
	Calcule le nombre d'échantillons à prélever pour l'ensemble de test.

	Args:
	- n_samples (int): Le nombre total d'échantillons.
	- test_size (float): Une proportion, ou un nombre entier, à utiliser.

	Returns:
	- int: Le nombre d'échantillons à prélever pour l'ensemble de test.
	"""
	if isinstance(test_size, int):
		return min(test_size, n_samples)
	elif 0 < test_size < 1:
		return int(test_size * n_samples)
	else:
		raise ValueError("'test_size' doit être un entier ∈ [1, n_samples] ou un flottant ∈ ]0, 1[ .")

class BasicSplitter(object):
	"""
	Classe de base pour les splitters du module.

	Attributes:
	- features (DataFrame): Les features utilisées.
	- labels (DataFrame): Les labels utilisés.
	- index (Index): Les index disponibles pour les données.
	- n_samples (int): Le nombre d'échantillons dans le jeu de données.
	- test_size (int): Le nombre d'échantillons dans l'ensemble de test.
	- test_split (List[int]): Une liste stockant les indices de l'ensemble de test.
	- train_split (List[int]): Une liste stockant les indices de l'ensemble d'entraînement.

	Methods:
	- get_features_labels(): Retourne les features et les labels.
	- train_test_split(): Divise les données en ensembles d'entraînement et de test.
	- _split(): Récupére les indices pour les ensembles d'entraînement et de test.
	"""

	def __init__(self, data: pd.DataFrame, labels: list, test_size: Union[float, int]) -> None:
		"""
		Initialise un objet BasicSplitter.

		Args:
		- data (DataFrame): Les données à utiliser.
		- labels (list): Les colonnes indiquant les labels à utiliser.
		- test_size (Union[float, int]): La taille de l'ensemble de test.

		Returns:
		- BasicSplitter: Un objet BaseSplitter.
		"""
		self.features: pd.DataFrame = data.drop(columns=labels, axis=1)
		self.labels: pd.DataFrame = data[labels]
		self.index: pd.Index = self.labels.index
		self.n_samples: int = data.shape[0]		
		self.test_size: int = calculate_test_samples(n_samples=self.n_samples, test_size=test_size)
		self.test_split: List[int] = []
		self.train_split: List[int] = []

	def get_features_labels(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""
		Retourne les features et les labels.

		Returns:
		- Tuple[DataFrame, DataFrame]: Un tuple contenant les features et les labels.
		"""
		return (self.features, self.labels)
	
	def train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		"""
		Divise les données en ensembles d'entraînement et de test.

		Returns:
		- Tuple[DataFrame, DataFrame, DataFrame, DataFrame]: Le tuple est composé de (x_train, x_test, y_train, y_test).
		"""
		x_train = self.features.loc[self.train_split]
		x_test = self.features.loc[self.test_split]
		y_train = self.labels.loc[self.train_split]
		y_test = self.labels.loc[self.test_split]
		return x_train, x_test, y_train, y_test
	
	def _split(self) -> None:
		"""
		Récupére les indices pour les ensembles d'entraînement et de test.

		Returns:
		- None
		"""
		pass