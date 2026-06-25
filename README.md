

# Detekcija pospanosti vozača 

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

Korišćeni modeli:
1. CNN (baseline)


```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc3 = nn.Linear(16 * 6 * 6, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.fc3(x)
```
2. ResNet18 (Transfer Learning)
```python   
class ResNetModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

    def forward(self, x):
        return self.model(x)
```
## Zašto su korišćeni i CNN i ResNet18?

U okviru ovog projekta korišćena su dva različita pristupa dubokom učenju: **custom CNN model** i **ResNet18 arhitektura sa transfer learning-om**.

### CNN (Custom model)

CNN model je implementiran kao **baseline rešenje** kako bi se:

- testirala osnovna sposobnost modela da prepozna obrasce na slikama
- dobila referentna tačnost bez korišćenja unapred treniranih mreža
- analizirala efikasnost jednostavne arhitekture u odnosu na kompleksnije modele

Ovaj model ima manji broj parametara, brže se trenira i koristi se kao početna tačka u poređenju performansi.

---

### ResNet18 (Transfer Learning)

ResNet18 je korišćen kao napredniji model zasnovan na **transfer learning-u**, gde se koristi mreža prethodno trenirana na ImageNet datasetu.

Razlog za njegovu upotrebu:

- bolja sposobnost ekstrakcije kompleksnih karakteristika
- stabilnije treniranje zahvaljujući residual connections
- bolje generalizacione performanse na manjim dataset-ima
- brže postizanje visokih rezultata u odnosu na training from scratch

---

- direktno poređenje jednostavnog i naprednog pristupa
- analizu uticaja arhitekture na performanse
- validaciju da transfer learning daje bolje rezultate u realnim uslovima
- 

---
*Primarni model koji se koristi i posmatra za ovaj zadatak je ResNet.*


# Trening modela

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

Najbolji modeli čuvaju se u datotekama:

```text
resnet_model.pth
```
```text
cnn_model.pth
```

---

# Analiza osetljivosti i hiperparametarska optimizacija

## 1. Learning Rate analiza

Testirane vrednosti za ResNet:

| Learning Rate | Accuracy |
| ------------- | -------- |
| 0.001         | 98.00%   |
| 0.0005        | 99.27%   |
| 0.0001        | 97.87%   |

Najbolji rezultat ostvaren je za:

```text
LR = 0.0005
```

Testirane vrednosti za CNN:

| Learning Rate | Accuracy | 
|--------------|----------|
| 0.001        | 97.13%   | 
| 0.0005       | 96.07%   | 
| 0.0001       | 91.87%   | 
---

## 2. Poređenje optimizatora

Testirani optimizatori:

| Optimizator    | Learning Rate |
| -------------- | ------------- |
| Adam           | 0.0005        |
| SGD + Momentum | 0.001         |

### Rezultati
ResNet

| Optimizator | Accuracy |
| ----------- | -------- |
| Adam        | 98.6%    |
| SGD         | 97.4%    |

CNN

| Optimizer | Accuracy | 
|----------|----------|
| Adam     | 93.53%   | 
| SGD      | 97.13%   | 

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

Rezultati na test skupu za ResNet:

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

Rezultati pokazuju da duboke neuronske mreže predstavljaju efikasno rešenje za automatsku detekciju pospanosti vozača i mogu poslužiti kao osnova za razvoj naprednih sistema za povećanje bezbednosti u saobraćaju

## License

This project is licensed under the MIT License - see the LICENSE file for details.
