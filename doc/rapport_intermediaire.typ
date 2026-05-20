// =====================================================================
//  Rapport intermédiaire : Interprétabilité du prédicteur phage-bactérie
// =====================================================================

#set document(
  title: "Rapport intermédiaire : Interprétabilité phage-bactérie",
  author: "Nathan Wulliamoz",
)
#set page(
  paper: "a4",
  margin: (top: 18mm, bottom: 18mm, left: 20mm, right: 20mm),
  numbering: "1 / 1",
)
#set text(size: 11pt, lang: "fr")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: none)

#show heading.where(level: 1): it => block(above: 1.4em, below: 0.6em,
  text(size: 1.4em, weight: "bold", it.body))
#show heading.where(level: 2): it => block(above: 1.1em, below: 0.4em,
  text(size: 1.15em, weight: "bold", it.body))
#show heading.where(level: 3): it => block(above: 0.9em, below: 0.3em,
  text(size: 1.0em, weight: "bold", it.body))
#show raw: set text(font: "DejaVu Sans Mono", size: 0.9em)
#show link: set text(fill: rgb("#1a4d80"))

// Tableaux : style booktabs (lignes horizontales seulement) + alignés à gauche
#show figure.where(kind: table): set align(center)
#set table(
  stroke: none,
  inset: (x: 8pt, y: 5pt),
)

// ---------- Page de titre ----------
#align(center)[
  #v(2cm)
  #text(size: 1.9em, weight: "bold")[Rapport intermédiaire]
  #v(0.3em)
  #text(size: 1.3em)[Interprétabilité du prédicteur d'interactions phage-bactérie]
]
#v(0.8cm)
#line(length: 100%, stroke: 0.5pt + gray)
#v(0.5em)

#outline(title: [Table des matières], depth: 3, indent: 1em)

#pagebreak()

// ===================================================================
= 1. Contexte et objectif

Le projet de base (PBIP) entraîne un classifieur binaire prédisant si un couple
(bactérie, phage) donne lieu à une infection, à partir d'embeddings produits
par plusieurs _foundation models_ d'ADN (NT2, MegaDNA, DNABERT2). Le modèle
complet de référence (`best`) concatène, pour chaque côté (phage / bactérie),
les embeddings produits par les 3 foundation models, chacun avec sa propre
*stratégie de merging* (truncate, top+bottom, max, TK-PERT…), avant de
passer le tout à un MLP classifieur.

L'objectif de ce travail est de *comprendre ce que ce modèle apprend* :

+ quels foundation models contribuent réellement à la décision ?
+ quelles dimensions de leurs embeddings sont décisives ?
+ quelles propriétés de la séquence ADN expliquent l'activation de ces
  dimensions, et donc la prédiction d'infection ?

Le tout dans un cadre d'*explicabilité post-hoc* (XAI).

#pagebreak()

// ===================================================================
= 2. Travaux réalisés

== 2.1 Importance des modeles de fondations
#text(size: 0.9em, fill: rgb("#555555"))[Notebook : `model_importances_analysis.ipynb`]

Afin de mieux comprendre le modèle et ses prédictions dans une optique
d'*explicabilité*, j'ai voulu réaliser une forme de *feature selection* au
niveau des foundation models : identifier ceux dont les embeddings portent
réellement le signal de la décision, et écarter ceux qui n'ajoutent que du
bruit.

Sur le modèle `best` étendu (NT2 + DNABERT2 + MegaDNA des deux côtés, *chaque
modèle découpé en variantes Top / Bottom* quand sa stratégie de merging
concatène first+last), j'ai mesuré la contribution de chaque (modèle, partie)
au MCC, par deux méthodes complémentaires :

=== 2.1.1 Valeurs de Shapley (contribution marginale moyenne)


#figure(
  image("images/shapley_values_models_splittet.png", width: 100%),
  caption: [Valeurs de Shapley par composant],
)



Lecture : les trois plus gros contributeurs au MCC sont :

+ *MegaDNA (bactérie)* : de loin le plus important (≈ 0.20)
+ *NT2 (phage, partie top)* : ≈ 0.15
+ *MegaDNA (phage, partie bottom)* : ≈ 0.06

Les autres composants ont des valeurs de Shapley proches de zéro. Cela suggère que le modèle `best` est sur-paramétré : il utilise beaucoup de dimensions qui n'apportent pas d'information utile, et qui peuvent même ajouter du bruit.

#pagebreak()

=== 2.1.2 Ablations exhaustives

En complément, j'ai ablaté des combinaisons de 5 modèles sur les 8 (ce qui
isole chaque triplet restant) et j'ai mesuré le ΔMCC :

#figure(
  image("images/5_model_ablatés_splitted.svg", width: 100%),
  caption: [Ablations à 5 modèles, top 20 sur 56 combinaisons],
)

Les pires dégradations (ΔMCC ≈ −0.28) sont les ablations qui *retirent
MegaDNA-bact ET NT2-phage-top*. Les ablations qui retirent DNABERT2 ne
dégradent pas (et améliorent parfois). Cohérent avec les Shapley.

Plus frappant : certaines configurations qui *retirent 5 composants sur les
8* obtiennent un *MCC légèrement supérieur* à celui du modèle complet
(ΔMCC ≈ +0.02). Autrement dit, garder seulement 3 composants suffit, et le
modèle `best` est donc sur-paramétré : enlever les composants bruyants n'est
pas neutre, c'est même bénéfique.

*Conclusion 2.1* : MegaDNA-bact et NT2-phage-top sont les piliers.
MegaDNA-phage-bottom est candidat sérieux comme troisième contributeur.
DNABERT2 est négligeable.

#pagebreak()

== 2.2 Validation MegaDNA-phage-bottom
#text(size: 0.9em, fill: rgb("#555555"))[Notebook : `best_small_comparison.ipynb`]

Les valeurs de Shapley suggèrent que MegaDNA-phage-bottom apporte un petit
gain (≈ 0.06 de MCC marginal). Mais ces valeurs sont calculées sur un seul
training : il faut vérifier que ce gain résiste au bruit de l'entraînement.

Procédure : deux architectures candidates entraînées *10 fois chacune* sur
le split PredPhi train, évaluées sur deux jeux de test :

- *best_small2* : MegaDNA (bact, Truncate) + NT2 (phage, Truncate)
- *best_small3* : best_small2 + MegaDNA (phage, BottomTruncate)

#figure(
  image("images/small2_small3_comp.png", width: 100%),
  caption: [best_small2 vs best_small3 sur 10 runs],
)

#figure(
  table(
    columns: 4,
    align: (left, left, center, center),
    table.hline(stroke: 1pt),
    table.header[*Test*][*Métrique*][*best_small2*][*best_small3*],
    table.hline(stroke: 0.5pt),
    [PredPhi],              [MCC],       [0.55 ± 0.03], [0.55 ± 0.03],
    [PredPhi],              [F1],        [0.77 ± 0.02], [0.78 ± 0.02],
    [Généralisation (all)], [MCC],       [0.33 ± 0.03], [0.33 ± 0.02],
    [Généralisation (all)], [F1],        [0.58 ± 0.01], [0.57 ± 0.01],
    [Généralisation (all)], [Précision], [0.43 ± 0.01], [0.42 ± 0.01],
    [Généralisation (all)], [Recall],    [0.88 ± 0.02], [0.93 ± 0.02],
    table.hline(stroke: 1pt),
  ),
  caption: [Métriques moyennes ± std sur 10 runs],
)

*Constat* : les deux configurations sont *essentiellement équivalentes* sur
le MCC, à la fois sur PredPhi (0.55 vs 0.55) et sur la généralisation `all`
(0.33 vs 0.33), avec un écart bien inférieur à l'écart-type entre runs. La
seule différence systématique : `best_small3` *gagne sur le recall* en
généralisation (0.93 vs 0.88), mais *au prix de la précision* (0.42 vs 0.43)
— c'est-à-dire un modèle plus biaisé vers le positif, sans réel gain
discriminatif.

*Conclusion 2.2* : l'ajout de MegaDNA-phage-bottom n'apporte pas de gain
robuste. *Décision pour la suite : on garde uniquement MegaDNA (bact) + NT2
(phage)* = `small2`, plus simple et avec un MCC équivalent. Cette
architecture est utilisée pour toute la suite des analyses XAI.

#pagebreak()

== 2.3 Réduction dimensionnelle guidée par IG
#text(size: 0.9em, fill: rgb("#555555"))[Notebook : `ig_dim_analysis.ipynb`]

Maintenant que le modèle est ramené à *deux* foundation models (un côté
bactérie : MegaDNA, un côté phage : NT2), on peut descendre d'un niveau et
chercher, à l'intérieur de leurs embeddings, *quelles dimensions* portent
réellement le signal de la décision. C'est une seconde passe de feature
selection, cette fois au grain plus fin.

Sur `small2`, j'ai calculé les *Integrated Gradients* par sample (Captum,
n_steps=50, baseline = embedding médian) pour identifier les dimensions qui
portent l'information.

Pour choisir le seuil de coupure, j'ai balayé plusieurs valeurs entre 5 % et
100 % d'IG cumulé, et pour chaque seuil j'ai réentraîné 10 fois un
classifieur sur les dims sélectionnées :

#figure(
  image("images/small2_dim_modifs.png", width: 90%),
  caption: [MCC sur le test set en fonction du % d'IG cumulé retenu (10 runs par point)],
)

*Lecture* : le MCC monte rapidement entre 5 % et 30 % d'IG retenu, puis
plateau au-delà. À partir de *30 %* le score est déjà au pic (≈ 0.56) et
n'augmente plus en gardant davantage de dims (et redescend même légèrement
à 70–80 %). C'est le *meilleur compromis performance / nombre de dimensions* :
on garde tout le signal utile sans le bruit ajouté par les dims peu
informatives.

*Résultat* : à *30 % d'IG cumulé*, j'ai retenu *69 dims bact + 125 dims
phage = 194 dims* (au lieu de 1732, soit ≈ 11 % de `small2`). Le MCC est
resté équivalent au modèle complet. Le modèle réduit a été sauvegardé sous
`output/pbip/small2/ig30/`.

Pour mettre l'ordre de grandeur en perspective vis-à-vis du modèle `best`
initial :

#figure(
  table(
    columns: 5,
    align: (left, left, left, center, center),
    table.hline(stroke: 1pt),
    table.header[*Modèle*][*Bact*][*Phage*][*Total*][*% de `best`*],
    table.hline(stroke: 0.5pt),
    [`best`],
      [NT2 (768) + MegaDNA (964) + DNABERT2 (768) = *2 500*],
      [NT2 (1 536) + MegaDNA (1 928) + DNABERT2 (12 288) = *15 752*],
      [*18 252*], [100 %],
    [`small2`],      [MegaDNA = *964*], [NT2 = *768*], [*1 732*], [*≈ 9.5 %*],
    [`small2_30ig`], [69],              [125],         [*194*],   [*≈ 1 %*],
    table.hline(stroke: 1pt),
  ),
  caption: [Cascade de réduction dimensionnelle],
)

Soit une réduction d'un facteur *≈ 94×* entre `best` et `small2_30ig`, sans
perte de MCC sur le set de test.

#pagebreak()

== 2.4 Évaluation de la généralisation
#text(size: 0.9em, fill: rgb("#555555"))[Notebook : `models_comparison.ipynb`]

Maintenant que le modèle a été *fortement réduit* (de 18 252 à 194 dimensions,
soit ≈ 1 % du `best` initial), il faut vérifier que cette compression n'a
rien sacrifié. Deux questions à trancher :

+ Les performances *en moyenne* (sur plusieurs runs, pour neutraliser le
  bruit du training) restent-elles équivalentes au modèle complet ?
+ Le modèle réduit *généralise-t-il* aussi bien hors-distribution ?

Pour y répondre, j'entraîne *10 fois chaque architecture* (`best`, `small2`,
`small2_30ig`) sur le train PredPhi, puis j'évalue sur deux jeux de test :
PredPhi-test (in-distribution) et le dataset externe `all` (7720 couples,
out-of-distribution). On compare ensuite *moyenne ± std* sur les 10 runs.

#figure(
  image("images/comparaison_best_smalls.png", width: 100%),
  caption: [Comparaison des 3 modèles],
)

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    table.hline(stroke: 1pt),
    table.header[*Modèle*][*MCC*][*F1*][*Accuracy*][*Précision*][*Rappel*],
    table.hline(stroke: 0.5pt),
    [`best`],
      [*−0.17 ± 0.04*], [0.26 ± 0.01], [0.45 ± 0.04], [0.23 ± 0.01], [0.30 ± 0.03],
    [`small2`],
      [*0.31 ± 0.02*],  [0.57 ± 0.01], [0.56 ± 0.02], [0.42 ± 0.01], [0.90 ± 0.02],
    [`small2_30ig`],
      [*0.31 ± 0.03*],  [0.57 ± 0.01], [0.57 ± 0.03], [0.42 ± 0.02], [0.89 ± 0.02],
    table.hline(stroke: 1pt),
  ),
  caption: [Métriques sur le dataset `all` (moyenne ± std sur 10 runs)],
)

*Résultat* : le modèle `best` (le plus gros) *ne généralise pas* sur le
dataset externe (MCC négatif, équivalent à pire qu'aléatoire). Le modèle
`small2`, avec seulement 2 foundation models, généralise nettement mieux.
Cohérent avec 2.1 : DNABERT2 ajoute du bruit qui sur-spécialise le modèle
au split d'entraînement PredPhi. Le modèle réduit `small2_30ig` (194 dims) 
fait aussi bien que `small2` (1732 dims) : les 194 dims retenues portent
l'essentiel du signal de généralisation.

#pagebreak()

== 2.5 Analyse XAI fine sur le modèle réduit (en cours)
#text(size: 0.9em, fill: rgb("#555555"))[Notebook : `ig_dim_analysis_30ig.ipynb`]

Maintenant que le modèle a été ramené à 194 dimensions effectives sans perte
de performance, on dispose d'une *surface d'attribution beaucoup plus
réduite* à analyser : chaque dimension restante est, par construction, une
de celles qui portent réellement le signal. C'est le bon moment pour passer
à une *explicabilité plus précise*, qui aurait été beaucoup plus difficile
à mener sur les 1732 dimensions de `small2` (signal noyé dans le bruit) ou
sur les 18 252 du modèle `best` complet (intractable et trompeur).

Sur le modèle final `small2_30ig` (194 dims), j'ai recalculé les IG,
*cette fois signés et ciblant la classe `infection`* (auparavant je regardais
`|IG|`, ce qui mélangeait pro et anti).

=== a. Dims pro-infection et anti-infection

- *Pro-infection* : dims avec IG signé moyen sur les *TP* le plus positif,
  c'est-à-dire des dims dont la valeur observée sur les vrais positifs pousse
  le modèle à dire « infection ».
- *Anti-infection* : dims avec IG signé moyen sur les *TN* le plus négatif,
  c'est-à-dire des dims dont la valeur observée sur les vrais négatifs fait
  baisser prob(infection).

#figure(
  image("images/IG_TP_TN.png", width: 100%),
  caption: [Top 20 dimensions pro-infection (gauche, IG signé moyen sur les TP) et
    anti-infection (droite, IG signé moyen sur les TN), côté bactérie (haut) et
    phage (bas)],
)

*Lecture* : les barres bleues / oranges (gauche) montrent les dims qui *poussent*
le modèle vers la classe « infection » sur les vrais positifs, les barres rouges /
violettes (droite) montrent celles qui *retiennent* le modèle vers la classe
« non-infection » sur les vrais négatifs.

#pagebreak()

=== b. Dims hautement discriminatives

Certaines dims apparaissent *dans les deux classements* (top pro *et* top
anti) : ex. bact dim 31 a `IG_TP = +0.053` et `IG_TN = −0.065`. Ça signifie
que la *valeur* de la dim diffère systématiquement entre samples positifs et
négatifs, et que le modèle s'appuie dessus comme axe discriminatif principal.

*Exemple, bact dim 31* : l'histogramme (gauche) et le boxplot (droite)
montrent la distribution de la valeur de la dim selon l'outcome.

#figure(
  image("images/distribution_dim31.png", width: 100%),
  caption: [Distribution de bact dim 31 par outcome],
)

La séparation TP / TN n'est *pas franche visuellement* sur l'histogramme
(les deux distributions se chevauchent largement), mais on observe quand
même une *tendance nette* : davantage de TP dans la partie gauche
(valeurs négatives), et les *médianes des boxplots* le confirment, TP
≈ −1.2 contre TN ≈ −0.3. C'est attendu : une dim seule ne suffit
généralement pas à trancher, c'est la *combinaison* de plusieurs dims
discriminatives qui produit la décision finale.

À noter sur le boxplot : les distributions s'alignent sur le *vrai label*
plutôt que sur la prédiction. TP et FN (label=1) ont des médianes proches
et plus basses ; TN et FP (label=0) plus hautes. Les erreurs (FP, FN) sont
concentrées dans la *zone ambigue* au milieu, là où la dim seule ne
tranche pas, et où d'autres dims peuvent renverser la décision.

*Limite biologique importante* : analyser bact dim 31 (ou n'importe quelle
dim) *isolément* ne fait pas réellement sens biologiquement. L'infection
est une *propriété conjointe* du couple bact × phage .
L'IG donne des attributions marginales par dim de chaque
côté, mais le vrai signal pro-infection est probablement dans
*l'interaction* entre certaines dims bact et certaines dims phage. Ce que
l'analyse univariée ci-dessus identifie comme « dim pro-infection » est
une approximation : la vraie réponse demande une analyse jointe (voir
section 5.3).

Toutefois, *toutes les bactéries ne sont pas égales face à l'infection* :
certaines sont très permissives et sont infectées par beaucoup de phages, tandis
que d'autres sont quasi inattaquables. Une dim bact pro-infection comme
dim 31 pourrait donc en partie *encoder l'identité* d'une bactérie
fortement susceptible.

#pagebreak()

=== c. Waterfall IG par sample

Pour comprendre *un cas précis* plutôt qu'une moyenne, j'ai tracé pour 4
samples représentatifs (TP / TN / FP / FN les plus confiants) la contribution
signée de chaque dim. Cela permet de voir :

- quels couples bact × phage activent fortement les dims pro-infection
- où se situent les faux positifs : quelles dims les ont trompés
- l'équilibre entre contribution bactérie vs phage par sample

*Exemple sur un TP très confiant* (sample idx=1062, prob(infection)=0.996) :

#figure(
  image("images/waterfall_TP.png", width: 100%),
  caption: [Waterfall IG pour un TP],
)

On retrouve bien la *dim 31 (orig 414)* déjà identifiée comme top
pro-infection dans la section 2.5 b : sa contribution IG est ici clairement
positive (≈ +0.04) et tire la prédiction vers l'infection. Les dims 35, 1 et
33 (toutes dans le top pro-infection global) tirent aussi dans le même sens.
Côté phage, les top dims pro-infection identifiées (92, 90, 33) sont
également présentes et positives.

#pagebreak()

// ===================================================================
= 3. Synthèse des résultats

+ *Sur-paramétrisation néfaste* : `best` (3 foundation models) a des
  résultats *à peine inférieurs* à `small2` sur le test set PredPhi
  (in-distribution, scores très proches), mais s'effondre complètement en
  généralisation sur `all` (MCC fortement négatif). `small2` tient le coup
  dans les deux cas et généralise nettement mieux.
+ *Choix d'architecture validé* : l'ajout d'une 3ème variante d'embedding
  (MegaDNA-phage-bottom) n'apporte pas de gain robuste sur 10 runs, on s'en
  tient à MegaDNA-bact + NT2-phage.
+ *Compressibilité forte* : le modèle complet `best` utilisait *18 252
  dimensions* au total (2500 bact + 15 752 phage, dont DNABERT2-phage à lui
  seul = 12 288). `small2` n'en garde que *1732* (≈ 9.5 % du total), et
  `small2_30ig` réduit encore à *194 dims* (≈ 11 % de small2, et *≈ 1 % du
  modèle initial*). La majorité du signal exploitable est portée par
  quelques dizaines de dims.
+ *Asymétrie bact/phage* : les bactéries (MegaDNA) ont moins de dims
  retenues (69) mais avec des IG signés plus forts par dim (top dim bact =
  +0.05) que le phage (top dim phage = +0.03). Côté bact, la décision est
  plus concentrée.
+ *Dims discriminatives identifiées* : quelques dims (bact 31, 33, 35 ;
  phage 115, 87, 100) portent l'essentiel du signal pro/anti-infection.


// ===================================================================
= 4. Limites actuelles

- *NT2 et MegaDNA* n'ont pas été conçus pour être interprétables par
  dimension. Chaque dim mélange potentiellement plusieurs concepts
  biologiques. Une dim « pro-infection » n'a probablement pas une
  signification biologique unique.
- *Aucun papier* ne mappe les dimensions individuelles de NT2 ou MegaDNA à
  des features biologiques précises.


// ===================================================================


= 5. Prochaines étapes

== 5.1 Recherche de patterns ADN qui maximisent une dim

*Idée centrale* : pour une dimension donnée _d_ (par exemple bact dim 31 =
la plus pro-infection), trouver *quelles séquences ADN maximisent la valeur
de cette dim* dans l'embedding du foundation model.

Trois approches praticables :

- *Sliding window* : prendre une séquence bactérienne réelle,
  faire glisser une fenêtre de taille fixe, recalculer l'embedding, mesurer
  la chute de la dim _d_. Les fenêtres qui
  font le plus chuter sont celles qui _contiennent_ le motif responsable.
- *Sélection par dataset* : parmi les séquences bactériennes, identifier
  celles qui ont la valeur de dim _d_ la plus élevée, et faire une analyse
  de motifs classique sur leurs régions ADN.
- *Génération adverse* : optimiser une séquence ADN par rapport à la dim _d_
  via un algorithme génétique ou une descente de gradient.


== 5.2 Sparse Autoencoders sur les embeddings

Un problème connu des embeddings de modèles comme NT2 est la superposition : une même dimension encode plusieurs concepts biologiques à la fois (GC, longueur, motifs…), ce qui rend l'interprétation difficile. L'idée d'utiliser un Sparse Autoencoder (SAE) est de forcer le modèle à re-exprimer ces embeddings dans un dictionnaire plus grand, où chaque "feature" est idéalement monosémantique, c'est-à-dire associée à un seul concept.
En pratique, on entraînerait un SAE (dict size ≈ 512–4096) sur les embeddings de small2_30ig et on vérifierait si les features obtenues corrèlent mieux avec les annotations biologiques que les dimensions brutes. Cette approche n'est pas nouvelle : Anthropic l'a montrée sur des LLM textuels (Bricken et al., Towards Monosemanticity, 2023), et Decode-gLM (2025) l'a reproduite sur le Nucleotide Transformer avec des résultats encourageants [https://www.biorxiv.org/content/10.1101/2025.10.31.685860v1].
Pour les phages, ça reste à explorer, les génomes sont mal annotés, donc il est difficile de savoir à l'avance si une feature SAE correspondrait à quelque chose de biologiquement interprétable comme le tropisme hôte, ou simplement à un artefact du pré-entraînement.

== 5.3 Analyse jointe bact × phage

Jusqu'ici je regarde les dims pro-infection côté bact, puis côté phage,
chacune dans son coin. Or l'infection arrive quand un phage donné est
compatible avec une bactérie donnée : une dim bact « pro-infection » ne
veut probablement rien dire toute seule, elle compte surtout en combinaison
avec certaines dims phage.

Une piste simple à essayer : prendre les top dims de chaque côté (par
exemple les 10 plus pro-infection bact et les 10 plus pro-infection phage),
former toutes les paires possibles (bact_i, phage_j), et regarder sur les
TP si la valeur de bact_i × phage_j est plus élevée que sur les TN. Si oui,
c'est qu'il y a bien un effet d'interaction entre ces deux dims, et pas
juste une somme d'effets indépendants.

== 5.4 Reproduire l'analyse sur le `best_model` initial du projet

Au démarrage du projet, l'architecture de référence n'était pas le `best`
utilisé dans ce rapport, mais un autre `best_model` avec des *choix de
stratégie de merging différents* (autres combinaisons de truncate /
top+bottom / TK-PERT…) et *une couche de PCA en amont du classifieur* pour
réduire les embeddings concaténés avant le MLP.

Il serait intéressant de *rejouer toute la chaîne d'analyse XAI* (Shapley
et ablation des composants, IG par dim, sélection guidée par IG,
caractérisation des dims pro/anti-infection) sur ce `best_model` initial,
puis de vérifier si on arrive aux *mêmes conclusions* :

- est-ce que MegaDNA-bact et NT2-phage ressortent toujours comme piliers,
  ou est-ce que les stratégies de merging changent la donne ?
- est-ce que la concentration de l'IG dans ≈ 10 % des dims tient encore
  avec une couche de PCA en amont ?
- est-ce que les dims pro-infection identifiées correspondent (après
  remontée à travers la PCA) à des notions biologiques cohérentes avec ce
  qu'on a trouvé sur `small2` ?

Si les conclusions tiennent, c'est une preuve que les résultats sont robustes
au choix de l'architecture exacte. Sinon, cela indique que les conclusions
dépendent du pipeline et qu'il faut nuancer.

== 5.5 Cross-validation des attributions par SHAP

Lancer SHAP (DeepExplainer ou KernelSHAP) sur les mêmes samples que les
waterfall IG. Si les deux méthodes désignent les mêmes dims comme
pro-infection sur les mêmes samples, la robustesse est confirmée.


#pagebreak()

// ===================================================================
= 6. Livrables actuels

#table(
  columns: 2,
  align: (left, left),
  table.hline(stroke: 1pt),
  table.header[*Fichier*][*Description*],
  table.hline(stroke: 0.5pt),
  [`output/pbip/small2/`],
    [Modèle `small2` (MegaDNA bact + NT2 phage) + config],
  [`output/pbip/small2/ig30/`],
    [Modèle `small2_30ig` (194 dims) + indices des dims retenues],
  [`output/pbip/best_small3/`],
    [Modèle entraîné pour la comparaison 2.2 (variante avec MegaDNA-phage-bottom)],
  [`analysis/model_importances_analysis.ipynb`],
    [Shapley + ablations des composants],
  [`analysis/best_small_comparison.ipynb`],
    [small2 vs small3, 10 runs × 2 datasets],
  [`analysis/models_comparison.ipynb`],
    [Comparaison `best` vs `small2` vs `small2_30ig` sur `all`],
  [`analysis/ig_dim_analysis.ipynb`],
    [Réduction par IG vers small2_30ig],
  [`analysis/ig_dim_analysis_30ig.ipynb`],
    [Analyse XAI fine du modèle réduit (en cours)],
  table.hline(stroke: 1pt),
)


// ===================================================================
= 7. Travail à réaliser, dans l'ordre

+ *Analyse jointe bact × phage* : passer des attributions marginales à
  une vraie analyse d'interactions de paires de dims (cf. section 5.3).
+ *Reproduire l'analyse sur le `best_model` initial* (autres stratégies
  de merging + couche PCA) pour vérifier que les conclusions tiennent en
  dehors du pipeline `small2` (cf. section 5.4).
+ *Finaliser `ig_dim_analysis_30ig.ipynb`* : corrélations dim / ADN
  complètes, features supplémentaires (entropie, codon usage, sites de
  restriction).
+ *Recherche de patterns ADN qui maximisent une dim* : sliding window /
  occlusion sur quelques séquences (cf. section 5.1).
+ *POC Sparse Autoencoder* sur les embeddings + comparaison avec les
  dims brutes (cf. section 5.2).
+ *SHAP en cross-validation* des attributions IG sur quelques samples
  (cf. section 5.5).
+ *Rédaction finale du rapport de TB*.