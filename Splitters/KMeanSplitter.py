#!/bin/python3
#-*- coding:utf-8 -*-

# Importation des modules
import pandas as pd
import numpy as np
from typing import Optional, Union
from .BasicSplitter import BasicSplitter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

class KMeanSplitter(BasicSplitter):
	"""
	Classe du splitter KMeans.

	Attributes:
	- features (DataFrame): Les features utilisées.
	- labels (DataFrame): Les labels utilisés.
	- index (Index): Les index disponibles pour les données.
	- n_samples (int): Le nombre d'échantillons dans le jeu de données.
	- test_size (int): Le nombre d'échantillons dans l'ensemble de test.
	- test_split (List[int]): Une liste stockant les indices de l'ensemble de test.
	- train_split (List[int]): Une liste stockant les indices de l'ensemble d'entraînement.
	- n_clusters (int): Nombre de clusters à créer.
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
				 n_clusters: int = 8,
				 pca: Optional[int] = None,
				 metric: str = "euclidean",
				 random_state: Optional[int] = None) -> None:
		"""
		Initialise un objet KMeanSplitter.

		Args:
		- data (DataFrame): Les données à utiliser.
		- labels (list): Les colonnes indiquant les labels à utiliser.
		- test_size (Union[float, int]): La taille de l'ensemble de test.
		- n_clusters (int): Nombre de clusters à créer. (défaut = 8)
		- pca (Optional[int]): Réduction dimensionnelle, pour accélérer l'echantillonnage. (défaut = None)
		- metric (str): La métrique de distance à utiliser. (défaut = "euclidean")
		- random_state (Optional[int]): La seed pour la reproductibilité. (défaut = None)

		Returns:
		- KMeanSplitter: Un objet KMeanSplitter.
		"""
		super().__init__(data=data, labels=labels, test_size=test_size)
		self.n_clusters: int = n_clusters
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
		data = pd.concat(objs=[self.features, self.labels], axis=1)

		# Récupération des paramètres
		random_state = self.random_state
		indexes = self.index.tolist()
		n_clusters = self.n_clusters
		metric = self.metric

		# Réduction de dimensionnalité (PCA) si spécifiée
		if self.pca is not None:
			pca = PCA(n_components=self.pca, random_state=random_state)
			data = pca.fit_transform(X=data)
		else:
			# Conversion en tableau NumPy
			data = data.to_numpy()

		# Clustering K-Means
		kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
		clusters = kmeans.fit(X=data).cluster_centers_
		n_samples_by_clusters = int(self.test_size / n_clusters)

		for cluster in clusters:
			cpt = 0
			# Calcul des distances
			distances = cdist(XA=data, XB=[cluster], metric=metric).reshape(-1,)
			distances = list(zip(indexes, distances))

			while cpt != n_samples_by_clusters:
				# Trouver l'index du point le plus proche du centroïde
				index, distance = min(distances, key=lambda x: x[1])

				# Retirer l'index et la distance des listes
				distances.remove((index, distance))
				indexes.remove(index)

				# Supprimer la ligne correspondante dans le tableau de données
				data = np.concatenate((data[:index], data[index+1:]), axis=0)

				# Ajouter l'index à l'ensemble de test
				self.test_split.append(index)
				cpt += 1

		# Ajouter les indices restants à l'ensemble d'entraînement
		self.train_split.extend(indexes)