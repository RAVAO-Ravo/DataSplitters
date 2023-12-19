#!/bin/python3
#-*- coding:utf-8 -*-

# Importation des modules
import pandas as pd
import random as rd
from typing import Optional, Union
from .BasicSplitter import BasicSplitter

class RandomSplitter(BasicSplitter):
	"""
	Classe du splitter aléatoire.

	Attributes:
	- features (DataFrame): Les features utilisées.
	- labels (DataFrame): Les labels utilisés.
	- index (Index): Les index disponibles pour les données.
	- n_samples (int): Le nombre d'échantillons dans le jeu de données.
	- test_size (int): Le nombre d'échantillons dans l'ensemble de test.
	- test_split (List[int]): Une liste stockant les indices de l'ensemble de test.
	- train_split (List[int]): Une liste stockant les indices de l'ensemble d'entraînement.
	- random_state (int): La seed pour la reproductibilité.

	Methods:
	- get_features_labels(): Retourne les features et les labels.
	- train_test_split(): Divise les données en ensembles d'entraînement et de test.
	- _split(): Récupére les indices pour les ensembles d'entraînement et de test.
	"""

	def __init__(self, data: pd.DataFrame, labels: list, test_size: Union[float, int], random_state: Optional[int] = None) -> None:
		"""
		Initialise un objet RandomSplitter.

		Args:
		- data (DataFrame): Les données à utiliser.
		- labels (list): Les colonnes indiquant les labels à utiliser.
		- test_size (Union[float, int]): La taille de l'ensemble de test.
		- random_state (Optional[int]): La seed pour la reproductibilité. (Défaut = None)

		returns:
		- RandomSplitter: Un objet RandomSplitter.
		"""
		super().__init__(data=data, labels=labels, test_size=test_size)
		self.random_state: Optional[int] = random_state
		self._split()
	
	def _split(self) -> None:
		"""
		Récupére les indices pour les ensembles d'entraînement et de test.

		Returns:
		- None
		"""
		# Initialisation de la seed spécifiée
		rd.seed(self.random_state)

		# Mettre en place les éléments nécessaires
		n_samples = self.n_samples
		test_size = self.test_size

		# Création d'un masque booléen de sélection aléatoire
		mask = [True if i < test_size else False for i in range(n_samples)]
		rd.shuffle(x=mask)

		# Attribution des indices à l'ensemble d'entraînement et à l'ensemble de test
		self.test_split = self.index[mask]
		self.train_split = self.index[[True if b == False else False for b in mask]]