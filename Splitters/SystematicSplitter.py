#!/bin/python3
#-*- coding:utf-8 -*-

# Importation des modules
import pandas as pd
from collections import deque
from typing import Optional, Union, Any, List
from .BasicSplitter import BasicSplitter

def rotate_indexes(indexes: List[int], rotations: int) -> deque:
	"""
	Effectue un certain nombre de rotations vers la droite sur une liste.

	Args:
	- indexes (List[int]): La liste à faire pivoter.
	- rotations (int): Le nombre de rotations à effectuer.

	Returns:
	- deque: La liste après les rotations.
	"""
	# Convertit la liste en deque pour des rotations efficaces
	rotated_list = deque(iterable=indexes)

	# Effectue les rotations
	rotated_list.rotate(rotations)

	return rotated_list

class SystematicSplitter(BasicSplitter):
	"""
	Classe du splitter systématique.

	Attributes:
	- features (DataFrame): Les features utilisées.
	- labels (DataFrame): Les labels utilisés.
	- index (Index): Les index disponibles pour les données.
	- n_samples (int): Le nombre d'échantillons dans le jeu de données.
	- test_size (int): Le nombre d'échantillons dans l'ensemble de test.
	- test_split (List[int]): Une liste stockant les indices de l'ensemble de test.
	- train_split (List[int]): Une liste stockant les indices de l'ensemble d'entraînement.
	- sorting_column (Any): Colonne (features ou labels) à utiliser pour le trie des données.
	- rotations (int): Nombre de rotations de la liste des index, cela change le point de départ.
	- step (int): Le pas utilisé pour l'échantillonnage. 

	Methods:
	- get_features_labels(): Retourne les features et les labels.
	- train_test_split(): Divise les données en ensembles d'entraînement et de test.
	- _split(): Récupére les indices pour les ensembles d'entraînement et de test.
	"""

	def __init__(self, data: pd.DataFrame, labels: list, test_size: Union[float, int], sorting_column: Any, rotations: Optional[int] = None) -> None:
		"""
		Initialise un objet SystematicSplitter.

		Args:
		- data (DataFrame): Les données à utiliser.
		- labels (list): Les colonnes indiquant les labels à utiliser.
		- test_size (Union[float, int]): La taille de l'ensemble de test.
		- sorting_column (Any): Colonne (features ou labels) à utiliser pour le trie des données.
		- rotations (Optional[int]): Nombre de rotations de la liste des index, cela change le point de départ. (défaut = None)

		Returns:
		- SystematicSplitter: Un objet SystematicSplitter.
		"""
		super().__init__(data=data, labels=labels, test_size=test_size)
		self.sorting_column: Any = sorting_column
		self.rotations: int = rotations
		self.step: int = int(self.n_samples / self.test_size)
		self._split()

	def _split(self) -> None:
		"""
		Récupére les indices pour les ensembles d'entraînement et de test.

		Returns:
		- None
		"""
		# Récupèration des index
		indexes = pd.concat(objs=[self.features, self.labels], axis=1).sort_values(by=self.sorting_column).index

		# Récupèration du nombre de rotations
		rotations = self.rotations

		# Rotation de la liste des index, et transformation en deque, si cela est spécifié
		if rotations is not None:
			indexes = rotate_indexes(indexes=indexes, rotations=rotations)
		else:
			# Transformation en deque
			indexes = deque(indexes)

		# Variable itératrice
		i = 0

		# Création de l'ensemble de test
		while len(self.test_split) < self.test_size:
			# Récupération des index
			if i == 0:
				self.test_split.append(indexes.popleft())
			else:
				self.train_split.append(indexes.popleft())
			i += 1

			# Réinitialisation de 'i' lorsqu'il vaut 'step'
			if i == self.step:
				i = 0

		# Ajouts des index restants dans 'train_split'
		self.train_split.extend(indexes)