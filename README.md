1. Opis problema

Pospanost vozača predstavlja jedan od značajnih uzroka saobraćajnih nezgoda. Cilj ovog projekta je razvoj sistema za automatsku detekciju pospanosti na osnovu slika lica vozača korišćenjem dubokih neuronskih mreža.

Problem je formulisan kao zadatak binarne klasifikacije, gde model na osnovu ulazne slike određuje da li vozač pripada jednoj od dve klase:

DROWSY – vozač pokazuje znakove pospanosti
NATURAL – vozač je budan i u normalnom stanju

Za rešavanje problema korišćena je transfer learning tehnika zasnovana na ResNet18 arhitekturi prethodno treniranoj na ImageNet skupu podataka.

2. Podaci
Izvor podataka

Za trening i evaluaciju korišćen je javno dostupan dataset za detekciju pospanosti vozača koji sadrži slike lica podeljene u dve klase:

DROWSY
NATURAL

Dataset je organizovan u direktorijume:

Drowsy_dataset/
│
├── train/
│   ├── DROWSY/
│   └── NATURAL/
│
└── test/
    ├── DROWSY/
    └── NATURAL/
Struktura podataka

Svaka slika predstavlja lice osobe u frontalnom ili približno frontalnom položaju.

Model koristi slike rezolucije:

96 x 96 piksela
Preprocesiranje

Tokom treninga primenjene su sledeće transformacije:

Resize na 96×96
Konverzija u 3-kanalni format
Random Horizontal Flip
Random Rotation (±10°)
Color Jitter
Random Affine transformacija
Normalizacija korišćenjem ImageNet statistike
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

Za validacioni i test skup korišćene su samo determinističke transformacije:

Resize
Grayscale → RGB
Normalizacija

Augmentacija nije primenjivana nad test podacima.

Podela podataka

Trening skup je dodatno podeljen:

80% trening
20% validacija

Korišćen je fiksni random seed:

seed = 42

radi reproduktivnosti rezultata.

3. Arhitektura modela

Korišćen je model:

ResNet18

Model je učitan sa prethodno treniranim težinama:

models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

Poslednji klasifikacioni sloj zamenjen je slojem:

nn.Linear(
    in_features=512,
    out_features=2
)

koji odgovara binarnoj klasifikaciji.

Prednosti ResNet arhitekture
Duboka mreža sa residual vezama
Efikasno treniranje bez problema nestajanja gradijenta
Odlične performanse u zadacima klasifikacije slika
Pogodna za transfer learning
4. Trening modela
Funkcija greške

Korišćena je funkcija:

CrossEntropyLoss
Optimizator

Osnovni model koristi:

Adam

sa parametrima:

Learning rate = 0.0005
Batch size
16
Broj epoha

Maksimalno:

20
Early Stopping

Kako bi se sprečilo pretreniravanje modela korišćen je mehanizam ranog zaustavljanja:

patience = 4

Trening se prekida ukoliko validaciona tačnost ne pokazuje poboljšanje tokom četiri uzastopne epohe.

Najbolji model čuva se u datoteci:

resnet_model.pth
5. Analiza osetljivosti i hiperparametarska optimizacija
5.1 Analiza learning rate parametra

Testirane su sledeće vrednosti:

Learning Rate
0.001
0.0005
0.0001

Rezultati:

LR	Accuracy
0.001	98.00%
0.0005	99.27%
0.0001	97.87%

Najbolji rezultat ostvaren je za:

LR = 0.0005
5.2 Poređenje optimizatora

Testirani optimizatori:

Optimizator	LR
Adam	0.0005
SGD + Momentum	0.001

Rezultati:

Optimizator	Accuracy
Adam	98.6%
SGD	97.4%

Adam je pokazao bolje performanse i bržu konvergenciju.

5.3 ROC analiza

Za dodatnu evaluaciju korišćena je ROC kriva i AUC metrika.

ROC kriva prikazuje odnos između:

True Positive Rate
False Positive Rate

Veća površina ispod krive (AUC) ukazuje na bolju sposobnost razdvajanja klasa.

5.4 Confusion Matrix analiza

Korišćena je matrica konfuzije radi analize:

Tačno klasifikovanih uzoraka
Lažno pozitivnih predikcija
Lažno negativnih predikcija

Na osnovu matrice moguće je identifikovati tipove grešaka koje model najčešće pravi.

6. Rezultati evaluacije

Dobijeni rezultati na test skupu:

Metrika	Vrednost
Accuracy	97%
Precision (DROWSY)	99%
Recall (DROWSY)	95%
F1-score (DROWSY)	97%
Precision (NATURAL)	95%
Recall (NATURAL)	99%
F1-score (NATURAL)	97%

Classification report:

              precision    recall  f1-score

DROWSY           0.99      0.95      0.97
NATURAL          0.95      0.99      0.97

Accuracy                              0.97
7. Diskusija

Rezultati pokazuju da ResNet18 model veoma uspešno razlikuje pospane i budne vozače.

Najbolje performanse ostvarene su korišćenjem:

Adam optimizatora
Learning rate = 0.0005
ImageNet normalizacije

Primećeno je da korišćenje ImageNet normalizacije daje bolje rezultate od standardne normalizacije na opseg [-1, 1], što je očekivano zbog upotrebe prethodno treniranog ResNet modela.

Iako su ostvarene veoma visoke performanse, potrebno je uzeti u obzir da dataset može biti relativno homogen i da rezultati ne moraju u potpunosti odražavati performanse sistema u realnim uslovima vožnje.

Ograničenja
Evaluacija je zasnovana na pojedinačnim slikama.
Ne koriste se temporalne informacije iz video zapisa.
Model nije testiran u različitim uslovima osvetljenja.
Moguće je postojanje velike sličnosti između uzoraka u trening i test skupu.
8. Zaključak

U okviru projekta razvijen je sistem za detekciju pospanosti vozača zasnovan na ResNet18 konvolucionoj neuronskoj mreži.

Primena transfer learning pristupa omogućila je postizanje visokih performansi uz relativno kratko vreme treniranja. Najbolji model ostvario je tačnost veću od 97%, dok je tokom hiperparametarske optimizacije postignuta tačnost od preko 99%.

Rezultati pokazuju da duboke neuronske mreže predstavljaju efikasno rešenje za automatsku detekciju pospanosti vozača i mogu poslužiti kao osnova za razvoj naprednih sistema za povećanje bezbednosti u saobraćaju.
