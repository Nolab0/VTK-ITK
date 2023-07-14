# Mini Projet ITK/VTK
## Thibault BOUTET - Jean FECHTER

## Déroulement du projet
L'objectif de ce projet était d'effectuer une étude longitudinale d'une tumeur à partir de deux scans du même patient.
Le but d'une telle étude est de determiner l'évolution d'une tumeur afin de choisir des traitements adaptés.
Pour réaliser cette étude, nous avons suivi les étapdes suivantes:
<ul>
	<li>Recalage des deux scans</li>
	<li>Segmentation de la tumeur</li>
	<li>Analyse et visualisation des changements</li>
</ul>

Pour lancé le projet: `python vtk-itk.py`

Nos expérimentations et nos visualisations intérmédiaire se trouvent dans le fichier ***research.ipynb***.

### Données
Pour ce projet, nous avions a notre disposition deux scans au format ***.nrrd**.
La lecture de ces fichiers à été faite avec une méthode similaire à celle vue en TP.

Il est a noté que pour réaliser ce projet, nous nous sommes extensivement basés sur des affichages intermédiaire de la coupe 80 de la vue sagitale. Celle-ci ce situe environ à la moitié du scan et offre une bonne visualisation d'ensemble. Cette coupe a particulièrement été utile pour vérifier nos résultats des prochaines étapes.

![Alt text](image-1.png)
*Coupe 80 de la vue sagitale*

## Recalage
Etant donné que les deux scans du patient on été réalisés à deux dates différentes, nous devions les recaler entre-eux afin de pouvoir effectuer nos analyses.

Nous  avons choisis de recaler le scans ***case6_gre2.nrrd*** sur le sacan ***case6_gre1.nrrd***.

Pour ce faire, nous avons testé les méthodes de recalage suivantes:
<ul>
	<li>B-spline</li>
	<li>Rigide (translation)</li>
</ul>
Nous nous sommes rendu compte que le recalage rigide uniquement avec une translation donnait de très bon résulat de manière rapide. En effet, il semblerait que les deux scans diffèrent à une translation près.

![Alt text](image-2.png)
*Scan 1 | Scan 2 | Scan 2 recalé*

## Segmentation
L'objectif de cette partie est de séparé la tumeur cérébrale des autres parties de l'image. Pour ce faire, nous avons utilisé la même procédure que celle vu lors des séances de TP.

Afin d'obtenir de bon résultats, nous avons du déterminer les bons paramètres de seuils et de point de départ de notre segmentation. Nous avons utilisé les paramètres suivant pour la segmentation:
<ul>
	<li>Seed-x: 60</li>
	<li>Seed-y: 80</li>
	<li>Seuil bas: 600</li>
	<li>Seuil haut: 800</li>
</ul>

![Alt text](image-3.png)
*Scan2 recalée segmenté | Scan 1 segmenté*

## Analyse et visualisation
