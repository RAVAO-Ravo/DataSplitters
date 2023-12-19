#!/bin/python3
#-*- coding:utf-8 -*-

# Importation des modules
import pandas as pd
import random as rd
from typing import Optional, Union, Any, List
from .BasicSplitter import BasicSplitter

def create_strats(labels: pd.DataFrame, strat_column: Any, n_strats: int) -> pd.DataFrame:
	"""
	Crée des strates à partir d'une colonne de valeurs continues.

	Args:
	- labels (DataFrame): Le DataFrame contenant les labels.
	- strat_column (Any): Le nom de la colonne à utiliser pour la création des strates.
	- n_strats (int): Le nombre de strates souhaité.

	Returns:
	- Dataframe: Un dataframe contenant les strates.
	"""
	# Utilise la fonction cut() pour créer des strates en fonction de la colonne spécifiée
	return pd.cut(x=labels[strat_column], bins=n_strats, labels=False, precision=6, include_lowest=True).to_frame(name=strat_column)

def get_strat_indexes(labels: pd.DataFrame, strat_column: Any, strat: Any, sample: int) -> List[int]:
	"""
	Retourne une liste d'indices échantillonnés dans une strates.

	Args:
	- labels (DataFrame): Le DataFrame contenant les labels.
	- strat_column (Any): Le nom de la colonne utilisée pour la stratification.
	- strat (Any): La strat utilisée.
	- sample (int): La nombre d'échantillons souhaités.

	Returns:
	- List[int]: Une liste d'indices échantillonnés dans une strates.
	"""
	# Filtrer et récupérer les indices associés à la strate
	indexes = labels[labels[strat_column] == strat].index

	# Créer un masque booléen pour l'échantillonnage
	mask = [True if i < sample else False for i in range(len(indexes))]
	
	# Mélanger le masque
	rd.shuffle(x=mask)

	# Retourner les indices correspondant au masque
	return [index for index, b in zip(indexes, mask) if b == True]

class StratifiedSplitter(BasicSplitter):
	"""
	Classe du splitter stratifié.

	Attributes:
	- features (DataFrame): Les features utilisées.
	- labels (DataFrame): Les labels utilisés.
	- index (Index): Les index disponibles pour les données.
	- n_samples (int): Le nombre d'échantillons dans le jeu de données.
	- test_size (int): Le nombre d'échantillons dans l'ensemble de test.
	- test_split (List[int]): Une liste stockant les indices de l'ensemble de test.
	- train_split (List[int]): Une liste stockant les indices de l'ensemble d'entraînement.
	- strat_column (Any): Colonne des labels à utiliser pour la l'échantillonnage stratifié.
	- classification (bool): Indique si l'on souhaite utiliser de la regression ou de la classification.
	- n_strats (int): Nombre de strates à créer dans le cadre d'une regression.
	- random_state (int)): La seed pour la reproductibilité.

	Methods:
	- get_features_labels(): Retourne les features et les labels.
	- train_test_split(): Divise les données en ensembles d'entraînement et de test.
	- _split(): Récupére les indices pour les ensembles d'entraînement et de test.
	"""

	def __init__(self, 
			  	 data: pd.DataFrame, 
			  	 labels: list, 
			  	 test_size: Union[float, int], 
			  	 strat_column: Any, 
			  	 classification: bool = True, 
			  	 n_strats: int = 2, 
			  	 random_state: Optional[int] = None) -> None:
		"""
		Initialise un objet StratifiedSplitter.

		Args:
		- data (DataFrame): Les données à utiliser.
		- labels (list): Les colonnes indiquant les labels à utiliser.
		- test_size (Union[float, int]): La taille de l'ensemble de test.
		- strat_column (Any): Colonne des labels à utiliser pour l'échantillonnage stratifié.
		- classification (bool): Indique si l'on souhaite utiliser de la régression ou de la classification.
		- n_strats (int): Nombre de strates à créer dans le cadre d'une régression.
		- random_state (Optional[int]): La seed pour la reproductibilité. (Défaut = None)

		Returns:
		- StratifiedSplitter: Un objet StratifiedSplitter.
		"""
		super().__init__(data=data, labels=labels, test_size=test_size)
		self.strat_column: Any = strat_column
		self.classification: bool = classification
		self.n_strats: int = n_strats
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
		labels = self.labels
		strat_column = self.strat_column
		n_samples = self.n_samples
		test_size = self.test_size

		# Si c'est une régression, créer des strates pour les labels
		if not self.classification:
			labels = create_strats(labels=labels, strat_column=strat_column, n_strats=self.n_strats)

		# Calcul des échantillons stratifiés en fonction de la taille de l'ensemble de test spécifiée
		strat_samples = dict(labels[strat_column].value_counts() / n_samples)
		strat_samples = {strat: int(test_size * pct) for strat, pct in strat_samples.items()}

		# Pour chaque strate, récupère les indices correspondant à l'ensemble de test
		for strat, sample in strat_samples.items():
			indexes = get_strat_indexes(labels=labels, strat_column=strat_column, strat=strat, sample=sample)
			self.test_split.extend(indexes)

		# Les indices restants vont dans l'ensemble d'entraînement
		self.train_split = [i for i in self.index if i not in self.test_split]