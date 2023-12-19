#!/bin/python3
#-*- coding:utf-8 -*-

# Importation des modules
import pandas as pd
from typing import Optional, Union
from .BasicSplitter import BasicSplitter
from .KennardStoneSplitter import max_min_distance_split
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

class SpxySplitter(BasicSplitter):
	"""
	Classe du splitter Kennard-Stone amélioré.

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
		Initialise un objet SpxySplitter.

		Args:
		- data (DataFrame): Les données à utiliser.
		- labels (list): Les colonnes indiquant les labels à utiliser.
		- test_size (Union[float, int]): La taille de l'ensemble de test.
		- pca (Optional[int]): Réduction dimensionnelle, pour accélérer l'echantillonnage. (défaut = None)
		- metric (str): La métrique de distance à utiliser. (défaut = "euclidean")
		- random_state (Optional[int]): La seed pour la reproductibilité. (défaut = None)

		Returns:
		- SpxySplitter: Un objet SpxySplitter.
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
		# Récupération des features, labels et métrique
		features = self.features.to_numpy()
		labels = self.labels.to_numpy()
		metric = self.metric

		# PCA si spécifiée
		if self.pca is not None:
			pca = PCA(n_components=self.pca, random_state=self.random_state)
			features = pca.fit_transform(X=features)

		# Calcul de la taille de l'ensemble d'entraînement
		train_size = self.n_samples - self.test_size

		# Vérification si la taille de l'ensemble d'entraînement est suffisante pour appliquer la méthode
		if train_size > 2:
			# Calcul des distances entre les points
			distances_features = cdist(XA=features, XB=features, metric=metric)
			distances_features = distances_features / distances_features.max()
			distances_labels = cdist(XA=labels, XB=labels, metric=metric)
			distances_labels = distances_labels / distances_labels.max()
			distances = distances_features + distances_labels

			# Appel de la fonction de division basée sur les distances maximales et minimales
			self.train_split, self.test_split = max_min_distance_split(distances=distances, indexes=self.index.tolist(), train_size=train_size)