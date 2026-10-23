# Modèle d'exécution de TorchTPU { #torchtpu-execution-model }

Lorsque vous exécutez un programme PyTorch avec TorchTPU, de nombreuses
opérations ont lieu en coulisses pour optimiser l'exécution. Comprendre ce
modèle vous aide à savoir comment ajuster votre utilisation et tirer parti de
certains points forts propres à TorchTPU.

## Pipeline d'exécution { #execution-pipeline }

### Création de l'opération différée { #create-deferred-op }

Lorsqu'une opération PyTorch (ou simplement « op ») est exécutée sur le
backend TorchTPU, aucun travail n'est effectué sur le TPU lui-même dans un
premier temps.

Le résultat logique du calcul est plutôt empaqueté dans une opération différée
(deferred op). Il s'agit d'un objet en mémoire côté CPU qui contient toutes
les informations nécessaires pour effectuer cette opération : ses entrées, les
étapes logiques à effectuer, ainsi que la forme et le type des sorties,
entre autres.

Le reste du pipeline, qui remplace cette opération différée symbolique par des
données réelles, s'appelle la matérialisation.

### Mise en file d'attente de l'opération sur un stream { #enqueue-op-on-a-stream }

TorchTPU prend en charge l'interface `torch.Stream`. Chaque fois qu'une
DeferredOp est créée, elle est placée dans le Stream actif courant afin d'être
mise en file d'attente pour une exécution ultérieure.

À ce stade, le thread appelant a terminé : tout ce qui suit se déroule de façon
asynchrone en arrière-plan (sauf en mode launch-blocking, dont nous parlerons
plus loin).

### Analyse du stream et préparation du plan d'exécution { #analyze-stream-and-prepare-execution-plan }

Dès qu'un stream contient du travail, un thread d'arrière-plan finit par
récupérer le travail de ce stream. Ce thread analyse alors l'ensemble du
travail présent dans le stream, en tenant compte de nombreux facteurs,
notamment :

*   En combien d'exécutions individuelles ce travail doit-il être divisé ?
*   Quelles sont les entrées et les sorties de chaque bloc d'exécution ?
*   Quel niveau d'optimisation de la compilation faut-il appliquer à chacun ?
*   Chaque unité de calcul a-t-elle déjà été vue, ou est-elle nouvelle ?
*   Sur quel cœur (TensorCore ou SparseCore) ce travail doit-il s'exécuter ?

En considérant le travail dans son ensemble, le thread d'arrière-plan peut
prendre de meilleures décisions d'exécution.

Une fois que le thread d'arrière-plan a identifié les sorties finales de
chaque bloc, il marque ces sorties comme en attente pour indiquer qu'elles
seront exécutées plus tard, puis passe à l'étape suivante.

### Compilation (ou récupération) des exécutables { #compile-or-fetch-executables }

Pour chaque étape d'exécution, le thread d'arrière-plan consulte un cache de
compilation pour savoir si un exécutable a déjà été compilé avec les bons
paramètres. Par défaut, il s'agit d'un cache en mémoire locale (niveau 1), avec
des options pour activer un cache persistant local à l'appareil (niveau 2)
ou des caches distants (niveau 3) afin de réduire au minimum le coût des
recompilations.

En cas de défaut de cache (cache miss), le graphe de compilation de l'unité est
converti en MLIR (principalement en StableHLO, sauf pour les kernels
personnalisés définis par l'utilisateur) et confié à un pool de threads de
compilation. Ce pool compile de façon asynchrone de nombreuses unités de
compilation en parallèle, et les renvoie au thread worker d'arrière-plan dès
qu'elles sont prêtes.

### Lancement de l'exécution { #start-execution }

Une fois les entrées rassemblées et l'exécutable préparé, une tâche
d'exécution est mise en file d'attente dans le runtime de XLA,
[PjRt](https://openxla.org/xla/pjrt).

PjRt renvoie des promesses pour cette future sortie. Elles sont ensuite
raccordées aux sorties pour indiquer que celles-ci sont en cours de
matérialisation.

### (Facultatif) Attente de la fin de l'exécution { #optional-wait-for-completion }

Pour savoir quand le travail est terminé, vous pouvez enregistrer un
`torch.Event` sur le stream et utiliser la méthode `torch.Event.query` pour
l'examiner.

Vous pouvez aussi, en complément ou à la place, bloquer sur le stream avec
n'importe quelle instruction de synchronisation (`torch.Event.synchronize`,
`torch.Stream.synchronize`, ou `torch.accelerator.synchronize` pour tout
l'appareil). Votre code Python est alors mis en pause jusqu'à ce que le TPU
signale la fin de l'exécution.

## Modes d'exécution { #execution-modes }

Le pipeline de programmation de TorchTPU est flexible et prend en charge
plusieurs modes d'exécution différents.

### Mode eager strict { #strict-eager-mode }

Aussi appelé « eager par défaut » ou « defer never », ce mode correspond à
l'expérience PyTorch habituelle, prête à l'emploi. Dans ce mode, chaque
opération ajoutée à la file d'attente signale immédiatement que le stream doit
être exécuté. Les opérations sont donc traitées une par une par le worker
d'arrière-plan de matérialisation.

Ces graphes d'exécution à une seule opération offrent un potentiel
d'optimisation minimal. Ils sont donc compilés avec le niveau de compilation
inférieur `-O1` pour maximiser la vitesse de compilation. Comme il y a de
nombreuses compilations individuelles, elles sont parallélisées très
efficacement sur le pool de threads de compilation, et le taux de réutilisation
des opérations individuelles est très élevé. Le surcoût de compilation reste
donc minimal pendant le « préchauffage » du cache de compilation.

Notez que, bien que ce mode exécute les opérations immédiatement et en
continu, il le fait toujours de façon asynchrone : une synchronisation reste
nécessaire pour attendre que le TPU ait terminé si vous voulez observer les résultats,
et les éventuelles erreurs d'exécution peuvent apparaître bien après
l'opération qui les a provoquées.

### Mode eager de débogage { #debug-eager-mode }

Aussi appelé « launch blocking », ce mode s'active en définissant la variable
d'environnement `TORCH_TPU_LAUNCH_BLOCKING=1`. Il est destiné uniquement au
débogage interactif, et non à la production.

Ce mode est identique au mode eager strict (exécution opération par
opération), sauf que l'hôte bloque et se synchronise à chaque opération.
L'exécution est ainsi artificiellement ralentie, mais les échecs sont signalés
dès qu'ils se produisent, ce qui aide à localiser précisément les erreurs.

### `torch.compile` { #torchcompile }

Pour des performances maximales, `torch.compile` offre à la fois un plus grand
potentiel d'optimisation et un surcoût d'exécution réduit.

Lorsque vous compilez une fonction ou un module précis avec `torch.compile`,
le backend de TorchTPU reçoit l'intégralité de la cible de compilation sous la
forme d'un [graphe FX](https://docs.pytorch.org/docs/stable/fx.html). Il
dispose ainsi d'un maximum d'informations sur l'exécution prévue, avec la
garantie qu'elle ne changera pas. Nous pouvons alors appliquer des
optimisations agressives, qui seraient trop risquées en mode eager.

En tant que backend en mode compilé, TorchTPU effectue un traçage initial (à
l'aide de tensors « placeholder », semblables à FakeTensor mais qui conservent
des métadonnées propres au TPU) et produit un graphe de calcul entièrement
différé qui reproduit le graphe FX. Une fois ce graphe créé, nous transmettons
ce programme StableHLO à XLA, qui le compile avec le niveau d'optimisation
supérieur `-O2`. Ce programme est ensuite empaqueté dans un callable Python et
renvoyé à l'utilisateur pour une exécution ultérieure.

Lorsque ce callable est appelé, la plupart des étapes de l'exécution eager sont
immédiatement court-circuitées. Aucune nouvelle DeferredOp n'est créée, aucune
analyse de graphe supplémentaire n'est nécessaire, et l'exécutable est déjà
fourni : aucune mise en cache n'est donc requise. L'exécutable est
immédiatement transmis à PjRt pour exécution, ce qui réduit au minimum le coût
côté hôte, avec un graphe de calcul entièrement optimisé par le compilateur
XLA.

### Mode eager optimisé { #optimized-eager-mode }

Aussi appelé mode « Defer and Fuse », il s'active en définissant
`TORCH_TPU_DEFER_AND_FUSE=1`. Il s'agit d'un nouveau mode hybride qui apporte
au mode eager de nombreux avantages de `torch.compile`.

Dans ce mode, la création d'une nouvelle DeferredOp ne déclenche pas
immédiatement l'exécution : les opérations restent différées entre les appels
Python, en attente dans la file du Stream.

La matérialisation commence lorsque l'une des deux conditions suivantes est
remplie :

*   L'utilisateur effectue une action qui force l'exécution. Les plus courantes
    sont le lancement d'un transfert de l'appareil vers l'hôte ou un appel
    explicite de synchronisation.
*   Nous détectons une répétition dans les opérations transmises par le
    dispatcher, ce qui indique que nous sommes dans une boucle et qu'il faut compiler une
    itération de cette boucle pour pouvoir la réutiliser facilement.

Cela déclenche une matérialisation en arrière-plan, comme en mode eager par
défaut, sauf que le graphe compilé est plus grand qu'une seule opération. Les
tensors déjà supprimés ou écrasés au moment où le signal de matérialisation est
émis sont ignorés. Ils sont généralement fusionnés dans les opérations en aval,
ou complètement éliminés si la sortie est abandonnée sans avoir été lue.
(Remarque : l'élimination du code mort est désactivée pour les opérations
collectives distribuées, afin d'éviter les interblocages entre TPU.)

Comme cette approche produit des graphes de compilation plus grands, nous
utilisons le niveau d'optimisation 2 (`-O2`) pour améliorer le temps
d'exécution. Comme nous compilons un plus petit nombre de graphes plus grands,
le pool de threads de compilation apporte moins de gain de parallélisation :
attendez-vous à des temps de compilation plus longs qu'en mode eager strict.

Sans être aussi optimisé que le mode `torch.compile`, le mode eager optimisé
améliore les performances jusqu'à 50 % par rapport au mode eager par défaut,
sans modification du code du modèle.

## Pièges à éviter { #pitfalls }

Ce pipeline d'exécution est optimisé à la fois pour la latence et pour le
débit, mais les performances peuvent se dégrader dans certains cas.

### Tous les modes : flux de contrôle côté Python et formes dépendantes des données { #all-modes-python-side-control-flow-and-data-dependent-shapes }

Dans PyTorch, il est facile de forcer par accident et sans s'en rendre compte
un transfert de l'appareil vers l'hôte. Une opération comme
« `if my_tensor.all()` » construit discrètement un tensor à valeur booléenne,
synchronise le stream et copie ce booléen vers le CPU avant de poursuivre le
flux de contrôle.

En mode eager par défaut, cela ne force aucune nouvelle rupture de graphe,
mais oblige l'ensemble du pipeline d'exécution du stream à se vider avant de
continuer. Pire encore, en mode eager optimisé ou en mode compilé, ce point de
synchronisation forcé provoque une rupture de graphe dans le stream fusionné en
mode eager ou dans le graphe FX, ce qui crée une barrière d'optimisation.

Un comportement similaire peut aussi se produire avec les opérations PyTorch
dont la forme de sortie dépend des données, comme `torch.masked_select`. Comme
la taille du tensor de sortie est inconnue tant que l'entrée du masque n'a pas
été évaluée, cette opération force elle aussi une synchronisation entièrement
bloquante du stream et un transfert des métadonnées de forme de l'appareil
vers l'hôte avant de continuer.

### Mode eager strict : les tout petits exécutables provoquent des microbulles { #strict-eager-mode-tiny-executables-cause-microbubbles }

La mise en file d'attente d'un exécutable dans PjRt prend un peu de temps :
généralement 50 microsecondes ou moins. Dans la plupart des cas, chaque
exécutable dure plus longtemps (même s'il ne dure que 1 ou 2 millisecondes) et
masque entièrement cette latence. On obtient alors un régime établi dans lequel
le TPU fonctionne en continu à son efficacité maximale.

En revanche, si trop d'exécutables mis en file d'attente durent moins que ce
seuil (par exemple en mode eager strict avec des tensors de petite taille), le
TPU peut manquer de travail. Lorsque le TPU s'exécute plus vite que le CPU ne
peut mettre du travail en file d'attente, l'exécution sur le TPU est
interrompue par de petites « microbulles » fréquentes, car la vitesse de
dispatch devient le goulot d'étranglement.

Pour éviter ce problème, assurez-vous que vos modèles effectuent suffisamment
de travail à chaque opération pour compenser ce petit surcoût de lancement, ou
utilisez le mode eager optimisé pour fusionner automatiquement les exécutions en
blocs de compilation à la volée.

### Mode compilé : vues comme arguments ou sorties { #compiled-mode-views-as-arguments-or-outputs }

Lors de l'appel d'un exécutable compilé, PjRt exige que les valeurs d'entrée
soient contiguës, sans décalage et d'un type de données précis. Si un argument
Tensor est une vue non triviale ou a été converti par bitcast depuis son type de
données d'origine, TorchTPU crée une copie temporaire du tensor en mode eager
(strict ou optimisé) pour appeler correctement l'exécutable compilé.

Pour éviter cette copie temporaire, assurez-vous que toutes les entrées d'une
fonction compilée sont contiguës. Si vous prévoyez d'appeler plusieurs
fonctions compilées à la suite, assurez-vous que les sorties des premières
fonctions sont contiguës, pour éviter les frictions liées au mode eager entre
les fonctions optimisées.

### Mode eager optimisé : conservation de tensors redondants { #optimized-eager-mode-retaining-redundant-tensors }

Dans tous les modes eager, les opérations s'exécutent dans l'ordre de leur
création et, chaque fois qu'un signal de matérialisation est émis, tous les
tensors encore actifs à ce moment-là sont matérialisés.

Si certains de ces tensors sont des tensors intermédiaires sans importance, le
mécanisme defer-and-fuse peut les identifier à tort comme des sorties
nécessaires et les matérialiser, ce qui peut réduire l'efficacité du graphe de
compilation.

Pour améliorer la fusion et réduire au minimum l'utilisation de la mémoire,
libérez les tensors dès qu'ils ne sont plus nécessaires, ou utilisez des
opérations sur place pour remplacer les tensors obsolètes par les valeurs
courantes.

## Points clés { #takeaways }

Le modèle d'exécution différée de TorchTPU offre un pipeline efficace, avec
différents compromis entre temps de compilation et temps d'exécution.

*   **Pour un temps de compilation minimal :** utilisez le mode eager strict
    (par défaut). Ce mode lance immédiatement l'exécution et parallélise la
    compilation pour obtenir le premier résultat le plus rapidement possible.
*   **Pour un débit maximal après préchauffage et une latence minimale :**
    utilisez `torch.compile`. Cette option élimine presque tout le surcoût
    du mode eager et maximise le potentiel d'optimisation du compilateur.
*   **Lorsqu'une exécution plus rapide est préférable mais que
    `torch.compile` n'est pas envisageable :** envisagez le mode eager optimisé.
*   **Pour le débogage :** utilisez le mode launch-blocking pour forcer le CPU
    et le TPU à avancer de façon synchronisée.
*   **Évitez les synchronisations forcées :** conservez un ensemble minimal de
    tensors contigus et utilisez des opérations à forte densité de calcul pour
    que le TPU dispose toujours de suffisamment de travail utile.
