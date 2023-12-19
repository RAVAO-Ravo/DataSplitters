#!/bin/python3
#-*- coding:utf-8 -*-

# Importation des modules
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, List
from .BasicSplitter import BasicSplitter
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

def max_min_distance_split(distances: np.ndarray, indexes: List[int], train_size: int) -> Tuple[List[int], List[int]]:
	"""
	Divise une liste d'indices en ensembles d'entraînement et de test basés sur les distances maximales et minimales.

	Args:
	- distances (np.ndarray): Matrice de distances entre les points.
	- indexes (List[int]): Liste d'indices à diviser.
	- train_size (int): Taille de l'ensemble d'entraînement désirée.

	Returns:
	- Tuple[List[int], List[int]]: Tuple contenant les indices de l'ensemble d'entraînement et de l'ensemble de test.
	"""
	index_train = []  # Liste pour stocker les indices de l'ensemble d'entraînement
	index_test = indexes  # Liste pour stocker les indices de l'ensemble de test

	# Sélection des deux points avec la plus grande distance
	point_one, point_two = np.unravel_index(indices=np.argmax(a=distances), shape=distances.shape)
	index_train.append(point_one)
	index_train.append(point_two)
	index_test.remove(point_one)
	index_test.remove(point_two)

	# Boucle pour compléter l'ensemble d'entraînement jusqu'à la taille désirée
	for _ in range(train_size - 2):
		# Sélection des distances entre les points déjà dans l'ensemble d'entraînement et le reste des points
		select_distance = distances[index_train, :]
		min_distance = np.min(a=select_distance[:, index_test], axis=0)
		max_min_distance = np.max(a=min_distance)

		# Sélection des points ayant la distance minimale maximale
		points = np.argwhere(select_distance == max_min_distance)[:, 1].tolist()

		for point in points:
			if point not in index_train:
				# Ajout du point à l'ensemble d'entraînement et suppression de l'ensemble de test
				index_train.append(point)
				index_test.remove(point)
				break

	return (index_train, index_test)

class KennardStoneSplitter(BasicSplitter):
	"""
	Classe du splitter Kennard-Stone.

	Attributes:
	- features (DataFrame): Les features utilisées.
	- labels (DataFrame): Les labels utilisés.
	- index (Index): Les index disponibles pour les données.
	- n_samples (int): Le nombre d'échantillons dans le jeu de données.
	- test_size (int): Le nombre d'échantillons dans l'ensemble de test.
	- test_split (List[int]): Une liste stockant les indices de l'ensemble de test.
	- train_split (List[int]): Une liste stockant les indices de l'ensemble d'entraînement.
	- pca (int): Réduction dimensionnelle, pour accélérer l'echantillonnage.
	- metric (str): La métrique de distance à utiliser.
	- random_state (int): La seed pour la reproductibilité.

	Methods:
	- get_features_labels(): Retourne les features et les labels.
	- train_test_split(): Divise les données en ensembles d'entraînement et de test.
	- _split(): Récupére les indices pour les ensembles d'entraînement et de test.
	"""
	
	def __init__(self, 
			  	 data: pd.DataFrame, 
				 labels: list, 
				 test_size: Union[float, int],
				 pca: Optional[int],
				 metric: str = "euclidean",
				 random_state: Optional[int] = None) -> None:
		"""
		Initialise un objet KennardStoneSplitter.

		Args:
		- data (DataFrame): Les données à utiliser.
		- labels (list): Les colonnes indiquant les labels à utiliser.
		- test_size (Union[float, int]): La taille de l'ensemble de test.
		- pca (Optional[int]): Réduction dimensionnelle, pour accélérer l'echantillonnage. (défaut = None)
		- metric (str): La métrique de distance à utiliser. (défaut = "euclidean")
		- random_state (Optional[int]): La seed pour la reproductibilité. (défaut = None)

		Returns:
		- KennardStoneSplitter: Un objet KennardStoneSplitter.
		"""
		super().__init__(data=data, labels=labels, test_size=test_size)
		self.pca: Optional[int] = pca
		self.metric: str = metric
		self.random_state: Optional[int] = random_state
		self._split()
	
	def _split(self) -> None:
		"""
		Récupére les indices pour les ensembles d'entraînement et de test.

		Returns:
		- None
		"""
		# Concaténation des features et labels pour obtenir les données
		data = pd.concat(objs=[self.features, self.labels], axis=1).to_numpy()

		# PCA si spécifiée
		if self.pca is not None:
			pca = PCA(n_components=self.pca, random_state=self.random_state)
			data = pca.fit_transform(X=data)

		# Calcul de la taille de l'ensemble d'entraînement
		train_size = self.n_samples - self.test_size

		# Vérification si la taille de l'ensemble d'entraînement est suffisante pour appliquer la méthode
		if train_size > 2:
			# Calcul des distances entre les points
			distances = cdist(XA=data, XB=data, metric=self.metric)

			# Appel de la fonction de division basée sur les distances maximales et minimales
			self.train_split, self.test_split = max_min_distance_split(distances=distances, indexes=self.index.tolist(), train_size=train_size)