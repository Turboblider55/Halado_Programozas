# Interaktív Neurális Háló – MNIST számfelismerés

Ez a projekt egy **NumPy-alapú többrétegű perceptron (MLP)** neurális hálót és egy hozzá tartozó **interaktív grafikus felületet (GUI)** tartalmaz.  
A cél: *kézzel rajzolt számok felismerése (0–9)* és a háló működésének vizuális megjelenítése.


## 🚀 Fő funkciók

- **MLP NumPy alapon** implementáció
- Két tanítási mód:
  - MNIST alapú batch training
  - Interaktív online tanítás (ha rosszat mond, kijavíthatod)
- ReLU rejtett rétegek, Softmax kimenet, cross-entropy loss
- Háló vizualizáció:
  - max **50 neuron/réteg**
  - aktiváció és súlyok színkódolva
- Nagy rajzfelület, valós idejű predikció
- `model.npz` fájlba menthető modellek
- Gyors MNIST tanítás (32k minta, 5 epoch)


## 💻 Telepítés és követelmények

**Python 3.8+** szükséges.

Könyvtárak:
- `numpy`
- `tkinter` (alapból telepítve)
- `gzip` (beépített)

MNIST fájlok szükségesek:
- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`


## 📁 Fájlstruktúra

```
project/
│
├── mlp_mnist.py      # A neurális háló és MNIST betöltő
├── gui_mlp.py        # Interaktív GUI + vizualizáció
└── model.npz         # Mentett modell (opcionális)
```


## ▶️ Használat

### 1. GUI futtatása
```bash
python gui_mlp.py
```

### 2. Rajzolás
A bal oldali rajzfelületre egérrel rajzolhatsz.  
A rajz automatikusan **28×28-as grayscale képpé** alakul.

### 3. Predikció
A középső panel mutatja:
- az aktuális előrejelzést
- a softmax valószínűségeket

### 4. Online tanítás
Ha a modell téved:
- kattints a „helyes szám” gombokra (0–9)
- a modell **azonnal tanul** egy lépést a rajzból

### 5. Gyors MNIST tanítás
A gomb:
- véletlenszerűen kiválaszt **32 000** mintát  
- **5 epoch** erejéig tanítja a modellt

### 6. Modell mentése/betöltése
- Mentés: `model.npz`
- Betöltés: automatikusan visszatölti a súlyokat


## 🧠 A neurális háló felépítése

**Alap architektúra:**

```
Input:  784 (28x28)
Hidden: 128 neuron (ReLU)
Hidden: 64 neuron (ReLU)
Output: 10 neuron (Softmax)
```

**Tanulási paraméterek:**
- Batch méret: `64`
- Loss: `cross-entropy`
- Tanulás: teljes backpropagation NumPy-al


## 🎛 Háló vizualizáció

A jobb oldali panel színkódolva jeleníti meg a háló működését:

### ⚪ Neuronok
- **szürke árnyalat**  
  - sötét → alacsony aktiváció  
  - világos → magas aktiváció  

### ⚓ Súlyok
- **zöld** → pozitív
- **piros** → negatív
- erősségük → szín intenzitás / vastagság

Max **50 neuron/réteg**, hogy gyors maradjon a GUI.


## ✍️ Online tanítás működése

1. Rajzolsz egy számot  
2. A modell prediktál  
3. Ha hibás → rákattintasz a helyes számra  
4. A modell egy lépést tanul (`fit_batch`)  

A változás **azonnal látszik** a vizualizáción.


## 📦 Modell mentése és formátuma

A mentett modell: `model.npz`

Tartalma:
```
layer0_W, layer0_b
layer1_W, layer1_b
layer2_W, layer2_b
...
```

Ez egy tömörített NumPy archívum.


## 🌱 Továbbfejlesztési lehetőségek

- több rejtett réteg hozzáadása  
- **dropout** integrálása overfitting ellen  
- konvolúciós réteg (CNN)  
- GPU-s verzió (TensorFlow / PyTorch)  
- loss-görbe megjelenítése GUI-ban  
- vizualizációs modul bővítése  
- tanulási ráta állítása GUI-ból  





