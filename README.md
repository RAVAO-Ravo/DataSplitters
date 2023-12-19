# DataSplitters

Le module **`Splitters`**, contenu dans ce repository, propose différentes techniques de splitting des données en ensembles d'entraînement et de test, une étape cruciale en apprentissage supervisé pour évaluer les modèles. Si l'échantillonnage aléatoire simple est souvent suffisant, certaines situations nécessitent des techniques plus avancées pour une bonne évaluation.

## Splitters

- Random : Échantillonnage aléatoire.
- Stratified : Échantillonnage aléatoire prenant en compte les stratifications des données.
- Systematic : Tri des données par rapport à une caractéristique, suivi d'un échantillonnage à un pas fixe (par exemple, prendre un échantillon tous les 3 éléments).
- KMeans : Création de clusters à partir des données, suivi d'un échantillonnage des points les plus proches de chaque centroïde.
- SPlit : Échantillonnage par utilisation de l'algorithme des Support Points.
- Kennard-Stone : Échantillonnage par utilisation de l'algorithme de Kennard-Stone.
- Spxy : Version dérivée de l'algorithme de Kennard-Stone.

Consultez le notebook `test_splitters.ipynb` pour des exemples d'utilisation des différents splitters implémentés.

D'autres techniques pourraient être implémentées dans le futur, et les méthodes existantes peuvent être améliorées (ou corrigées en cas d'erreurs).

## Téléchargement

### Installation du module

Pour récupérer le module, utilisez la commande suivante dans votre terminal :

```bash
git clone https://github.com/RAVAO-Ravo/DataSplitters.git
```

Assurez-vous d'avoir `Git` installé sur votre système avant d'exécuter cette commande.

### Installation des dépendances

Pour installer les dépendances requises, utilisez le fichier `requirements.txt`. Exécutez la commande suivante :

```bash
pip3 install -r requirements.txt
```

Assurez-vous d'avoir `python3` sur votre système.

### Implémentations d'origine

Certaines des implémentations utilisées dans ce module proviennent de sources externes. Voici les références pour ces implémentations :

- Kennard-Stone et Spxy: [https://hxhc.xyz/post/kennardstone-spxy/](https://hxhc.xyz/post/kennardstone-spxy/)

## Références

Le module **`Splitters`** utilise diverses techniques de splitting. Voici quelques-unes (liste non-exhaustive) des références utilisées pour développer ces méthodes :

- Kennard, R. W., & Stone, L. A. (1969). "Computer-aided design of experiments." Technometrics, 11(1), 137-148.

- Naes, T, T Isaksson, T Fearn, and T Davies. (2002). "Outlier Detection. A User-Friendly Guide to Multivariate Calibration and Classification." NIR Publications, Chichester.

- Reitermanova Z. (2010). "Data Splitting." WDS'10 Proceedings of Contributed Papers, Part I, 31–36.

- Joseph, V. R., & Vakayil, A. (2021). "SPlit: An Optimal Method for Data Splitting." Technometrics, 1-11.

## Licence

Ce projet est sous licence [Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

Vous êtes libre de :

- **Partager** : copier et redistribuer le matériel sous n'importe quel format ou médium.
- **Adapter** : remixer, transformer et construire à partir du matériel.

Selon les conditions suivantes :

- **Attribution** : Vous devez donner le crédit approprié, fournir un lien vers la licence et indiquer si des modifications ont été faites. Vous devez le faire d'une manière raisonnable, mais d'une manière qui n'implique pas que l'auteur vous approuve ou approuve votre utilisation du matériel.
- **ShareAlike** : Si vous remixez, transformez ou construisez à partir du matériel, vous devez distribuer vos contributions sous la même licence que l'original.

Veuillez consulter le fichier [LICENSE](LICENSE) pour plus de détails.