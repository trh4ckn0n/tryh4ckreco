# tryh4ckreco
# Face Pattern Generator - Anti-Reco Demo (by trhacknon)

Un outil pédagogique en Python permettant de générer et appliquer des motifs perturbateurs sur un visage, afin de démontrer la fragilité des systèmes de reconnaissance faciale basiques (par ex. HaarCascade d’OpenCV).

NOTE IMPORTANTE : Ce projet est uniquement destiné à un usage éducatif et académique.  
Il ne doit pas être utilisé dans des contextes réels ou pour contourner la sécurité.

------------------------------------------------------------

Fonctionnalités
- Détection faciale avant / après perturbation (OpenCV HaarCascade).
- Génération de motifs variés (damier, rayures, bruit, motifs fluo).
- Application automatique du motif sur une zone du visage (front, bouche, lunettes).
- Comparaison du nombre de visages détectés avant/après perturbation.
- Interface graphique (Tkinter) simple et stylisée, ambiance hacker-fluo trhacknon.
- Sauvegarde automatique des résultats (prof_original.png et prof_modifie.png).

------------------------------------------------------------

Installation

Clone le projet et installe les dépendances :

git clone https://github.com/ton-projet/facerec-pattern.git
cd facerec-pattern
pip install -r requirements.txt

Contenu du requirements.txt :

opencv-python
numpy
Pillow

------------------------------------------------------------

Utilisation

Mode script (CLI)
Exécuter simplement le script :

python main.py

Le programme :
1. Charge une image (prof.jpg par défaut).
2. Détecte le visage.
3. Applique un motif perturbateur.
4. Relance la détection.
5. Sauvegarde les deux versions (avant/après).

Mode GUI (interface graphique)

Une fenêtre Tkinter s’ouvre avec :
- Bouton Choisir une image.
- Bouton Sélectionner un motif.
- Bouton Appliquer le motif.
- Bouton Sauvegarder le résultat.

------------------------------------------------------------

Structure du projet

facerec-pattern/
├── main.py             # Script principal
├── motifgen.py         # Générateur de motifs
├── requirements.txt    # Dépendances Python
├── README.md           # Documentation du projet
└── prof.jpg            # Exemple d’image de test

------------------------------------------------------------

Exemple de résultat

Avant perturbation :  
prof_original.png

Après perturbation (motif fluo) :  
prof_modifie.png

------------------------------------------------------------

Avertissement

Ce projet est fourni uniquement à des fins éducatives.  
Son but est de montrer la fragilité des détecteurs faciaux simples dans un cadre académique.  

------------------------------------------------------------

Projet réalisé par trhacknon
