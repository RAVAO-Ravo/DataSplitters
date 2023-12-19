#!/bin/python3
#-*- coding:utf-8 -*-

# Importation des modules
import pandas as pd
from typing import Optional, Union
from .BasicSplitter import BasicSplitter
from twinning import twin

class SPlitSplitter(BasicSplitter):
	"""
	Classe du splitter Support Points.

	Attributes:
	- features (DataFrame): Les features utilisées.
	- labels (DataFrame): Les labels utilisés.
	- index (Index): Les index disponibles pour les données.
	- n_samples (int): Le nombre d'échantillons dans le jeu de données.
	- test_size (int): Le nombre d'échantillons dans l'ensemble de test.
	- test_split (List[int]): Une liste stockant les indices de l'ensemble de test.
	- train_split (List[int]): Une liste stockant les indices de l'ensemble d'entraînement.
	- start (int): Point de départ de l'algorithme Support Points.

	Methods:
	- get_features_labels(): Retourne les features et les labels.
	- train_test_split(): Divise les données en ensembles d'entraînement et de test.
	- _split(): Récupére les indices pour les ensembles d'entraînement et de test.
	"""

	def __init__(self, data: pd.DataFrame, labels: list, test_size: Union[float, int], start: Optional[int] = None) -> None:
		"""
		Initialise un objet SPlitSplitter.

		Args:
		- data (DataFrame): Les données à utiliser.
		- labels (list): Les colonnes indiquant les labels à utiliser.
		- test_size (Union[float, int]): La taille de l'ensemble de test.
		- start (Optional[int]): Point de départ de l'algorithme Support Points. (défaut = None)

		Returns:
		- SPlitSplitter: Un objet SPlitSplitter.
		"""
		super().__init__(data=data, labels=labels, test_size=test_size)
		self.start: Optional[int] = start
		self._split()

	def _split(self) -> None:
		"""
		Récupére les indices pour les ensembles d'entraînement et de test.

		Returns:
		- None
		"""
		# Concaténation des features et labels, puis conversion en tableau NumPy
		data = pd.concat(objs=[self.features, self.labels], axis=1).to_numpy()

		# Récupération des paramètres
		start = self.start
		n_samples = self.n_samples
		test_size = self.test_size

		# Calcul du nombre de fois à répéter la fonction twin
		r = int(1 / (test_size / n_samples))

		# Vérification de la validité du random_state
		if start is not None and (start < 0 or start >= n_samples):
			raise ValueError("'random_state' doit être un entier ∈ [0, len(features)-1[ .")

		# Appel de la fonction twin en utilisant random_state comme paramètre u1
		self.test_split = twin(data=data, r=r, u1=start).tolist()

		# Indices des échantillons d'entraînement sont ceux qui ne sont pas dans l'ensemble de test
		self.train_split = [i for i in self.index if i not in self.test_split]