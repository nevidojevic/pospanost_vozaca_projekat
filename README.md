

# Detekcija pospanosti vozača korišćenjem ResNet18 modela

## Opis projekta

Pospanost vozača predstavlja jedan od značajnih uzroka saobraćajnih nezgoda. Cilj ovog projekta je razvoj sistema za automatsku detekciju pospanosti na osnovu slika lica vozača korišćenjem dubokih neuronskih mreža.

Problem je formulisan kao zadatak **binarne klasifikacije**, gde model na osnovu ulazne slike određuje da li vozač pripada jednoj od dve klase:

| Klasa   | Opis                                |
| ------- | ----------------------------------- |
| DROWSY  | Vozač pokazuje znakove pospanosti   |
| NATURAL | Vozač je budan i u normalnom stanju |

Za rešavanje problema korišćena je **transfer learning** tehnika zasnovana na arhitekturi **ResNet18**, prethodno treniranoj na ImageNet skupu podataka.

---

# Podaci

## Izvor podataka

Korišćeni su i kombinovani javno dostupni datasetovi za detekciju pospanosti vozača sa platforme Kaggle, koji sadrži slike lica podeljene u dve klase:

* DROWSY
* NATURAL

### Struktura dataset-a

```text
Drowsy_dataset/
│
├── train/
│   ├── DROWSY/
│   └── NATURAL/
│
└── test/
    ├── DROWSY/
    └── NATURAL/
```

Svaka slika predstavlja lice osobe u frontalnom ili približno frontalnom položaju.

Ulazna rezolucija:

```text
96 × 96 piksela
```

---

## Preprocesiranje podataka

### Trening skup

Na trening podacima primenjene su sledeće transformacije:

* Resize (96×96)
* Grayscale → RGB (3 kanala)
* Random Horizontal Flip
* Random Rotation (±10°)
* Color Jitter
* Random Affine transformacija
* Konverzija u Tensor
* Normalizacija korišćenjem ImageNet statistike

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Validacioni i test skup

Na validacionom i test skupu korišćene su samo determinističke transformacije:

* Resize
* Grayscale → RGB
* ToTensor
* Normalize

Augmentacija nije primenjivana nad test podacima.

---

## Podela podataka

Trening skup je dodatno podeljen na:

* 80% trening
* 20% validacija

Radi reproduktivnosti rezultata korišćen je:

```python
seed = 42
```

---

# Arhitektura modela

Korišćen je model:

```python
models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)
```

Poslednji klasifikacioni sloj zamenjen je novim slojem sa dva izlaza:

```python
nn.Linear(
    in_features=512,
    out_features=2
)
```

što omogućava binarnu klasifikaciju između klasa DROWSY i NATURAL.

## Prednosti ResNet18 arhitekture

* Residual Connections
* Stabilno treniranje dubokih mreža
* Smanjen problem nestajanja gradijenta
* Visoke performanse u klasifikaciji slika
* Pogodna za transfer learning

---

# 🏋️ Trening modela

## Funkcija greške

```python
CrossEntropyLoss()
```

## Optimizator

Korišćen je:

```python
Adam
```

sa parametrima:

| Parametar     | Vrednost |
| ------------- | -------- |
| Learning Rate | 0.0005   |
| Batch Size    | 16       |
| Max Epochs    | 20       |

---

## Early Stopping

Kako bi se sprečilo pretreniravanje modela korišćen je mehanizam ranog zaustavljanja:

```python
patience = 4
```

Trening se prekida ukoliko validaciona tačnost ne pokazuje poboljšanje tokom četiri uzastopne epohe.

Najbolji model čuva se u datoteci:

```text
resnet_model.pth
```

---

# Analiza osetljivosti i hiperparametarska optimizacija

## 1. Learning Rate analiza

Testirane vrednosti:

| Learning Rate | Accuracy |
| ------------- | -------- |
| 0.001         | 98.00%   |
| 0.0005        | 99.27%   |
| 0.0001        | 97.87%   |

Najbolji rezultat ostvaren je za:

```text
LR = 0.0005
```

---

## 2. Poređenje optimizatora

Testirani optimizatori:

| Optimizator    | Learning Rate |
| -------------- | ------------- |
| Adam           | 0.0005        |
| SGD + Momentum | 0.001         |

### Rezultati

| Optimizator | Accuracy |
| ----------- | -------- |
| Adam        | 98.6%    |
| SGD         | 97.4%    |

Adam optimizator pokazao je bolju konvergenciju i više performanse.

---

## 3. ROC analiza

Za dodatnu evaluaciju korišćena je:

* ROC kriva
* AUC metrika

ROC kriva prikazuje odnos između:

* True Positive Rate (TPR)
* False Positive Rate (FPR)

Veća površina ispod krive (AUC) ukazuje na bolju sposobnost modela da razlikuje klase.

---

## 4. Confusion Matrix analiza

Matrica konfuzije korišćena je za analizu:

* Tačno klasifikovanih uzoraka
* Lažno pozitivnih predikcija
* Lažno negativnih predikcija

Na osnovu matrice moguće je identifikovati tipove grešaka koje model najčešće pravi.

---

# Rezultati evaluacije

Rezultati na test skupu:

| Metrika             | Vrednost |
| ------------------- | -------- |
| Accuracy            | 97%      |
| Precision (DROWSY)  | 99%      |
| Recall (DROWSY)     | 95%      |
| F1-score (DROWSY)   | 97%      |
| Precision (NATURAL) | 95%      |
| Recall (NATURAL)    | 99%      |
| F1-score (NATURAL)  | 97%      |

### Classification Report

```text
              precision    recall  f1-score

DROWSY           0.99      0.95      0.97
NATURAL          0.95      0.99      0.97

Accuracy                              0.97
```

---

# Diskusija

Rezultati pokazuju da ResNet18 model veoma uspešno razlikuje pospane i budne vozače.

Najbolje performanse ostvarene su korišćenjem:

* Adam optimizatora
* Learning Rate = 0.0005
* ImageNet normalizacije

Primećeno je da ImageNet normalizacija daje bolje rezultate od standardne normalizacije na opseg `[-1, 1]`, što je očekivano zbog korišćenja prethodno treniranog ResNet modela.

Iako su ostvarene veoma visoke performanse, potrebno je uzeti u obzir da dataset može biti relativno homogen i da rezultati ne moraju u potpunosti odražavati performanse sistema u realnim uslovima vožnje.

---

## Ograničenja

* Evaluacija je zasnovana na pojedinačnim slikama.
* Ne koriste se temporalne informacije iz video zapisa.
* Model nije testiran u različitim uslovima osvetljenja.
* Moguća je velika sličnost između trening i test uzoraka.

---

# Zaključak

U okviru projekta razvijen je sistem za detekciju pospanosti vozača zasnovan na **ResNet18** konvolucionoj neuronskoj mreži.

Primena transfer learning pristupa omogućila je postizanje visokih performansi uz relativno kratko vreme treniranja. Najbolji model ostvario je tačnost veću od **97%**, dok je tokom hiperparametarske optimizacije postignuta tačnost od preko **99%**.

Rezultati pokazuju da duboke neuronske mreže predstavljaju efikasno rešenje za automatsku detekciju pospanosti vozača i mogu poslužiti kao osnova za razvoj naprednih sistema za povećanje bezbednosti u saobraćaju.
