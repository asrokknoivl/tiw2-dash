
## M2-TIW Analyse de données-projet Dash

### Membres
- *Youssef AL AMRANI - M2 TIW*
- *M.Kais EL HADJ MUSTAPHA - M2 TIW*
- *Yara NEFAA - M2 TIW*

### Présentation du jeu de données
Le jeu de données choisi constitue une version élargie du jeu de données Goodbooks 10k, extrait à partir de l'API de la plateforme Goodreads. Le fichier books_enriched.csv intègre des champs additionnels. L'élément clé de cette nouvelle version réside dans l'inclusion d'un champ dédié à la description textuelle pour 9943 des 10 000 livres.

Nous avons opté pour cette version enrichie car elle comporte ainsi plusieurs opportunités diverses que ce soit en terme de nettoyage de données, d'exploitation de données, d'analyse, mais aussi de visualisation.

### Lancement du projet

- Avant de commencer, vérifiez que vous avez `git`, `python (version 3.x)` , `virtualenv` installés sur votre machine.
- Depuis la console de votre machine, lancez la commande `git clone https://github.com/asrokknoivl/tiw2-dash` et puis `cd tiw2-dash`
- Téléchargez les données avec le lien: https://www.mediafire.com/file/07h966kudgaoo8r/data.zip/file et décompressez les dans le repo que vous avez clone.
- Si les données sont créées dans un dossier *data* faites les sortir du dossier vers le dossier général (***/tiw2-dash***, ou se trouve le fichier `app.py`)
- Sur Linux, lancez la commande `python3 -m venv venv`
- Puis `source venv/bin/activate`,dans la console, le nom de l'environnement que vous avez choisi (ici, *venv*) devrait s'afficher entre parenthèses au début de la ligne.
- Tapez `pip install -r requirements.txt` pour installer toutes les dépendances.
- Si toutes les étapes précédences ont été exécutées correctement, vous pouvez lancer la commande `python3 app.py` qui devrait lancer une application web dash qui montrerait ce que nous avons fait dans ce projet.

Si vous n'arrivez pas à lancer l'app sur votre machine, nous l'avons déployé sur une VM depuis la plateforme *openStack* , vous pouvez y avoir accés à travers ce lien, http://192.168.246.62/.

Merci !
