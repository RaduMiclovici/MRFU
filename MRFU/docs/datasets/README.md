##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Set de imagini 
gi* **Modul de achiziție:** ☐ Senzori reali / ☐ Simulare / ☑ Fișier extern / ☐ Generare programatică
* **Perioada / condițiile colectării:** Noiembrie 2025 - decembrie 2025, condiții variate de iluminare

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 50
* **Număr de caracteristici (features):** 6
* **Tipuri de date:** ☐ Numerice / ☐ Categoriale / ☐ Temporale / ☑ Imagini
* **Format fișiere:** ☐ CSV / ☐ TXT / ☐ JSON / ☑ PNG / ☐ Altele: [...]

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| british_shorthair | image | - | [poze cu british shorthair] | 0–20 |
| labrador | image | – | [poze cu labradori] | 0-20|
| maine_coon | image | – | [poze cu maine coon] | 0-20|
| pug | image | – | [poze cu pug] | 0-20|
| siamese | image | – | [poze cu siamese] | 0-20|
| husky | image | – | [poze cu husky] | 0-20|

**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Distribuții pe caracteristici** (histograme)
* **Identificarea outlierilor** (IQR / percentile)

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă** (% pe coloană)
* **Detectarea valorilor inconsistente sau eronate**
* **Identificarea caracteristicilor redundante sau puternic corelate**

### 3.3 Probleme identificate

* [exemplu] Feature X are 8% valori lipsă
* [exemplu] Distribuția feature Y este puternic neuniformă
* [exemplu] Variabilitate ridicată în clase (class imbalance)

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor**
* **Tratarea valorilor lipsă:**
  * Feature A: imputare cu mediană
  * Feature B: eliminare (30% valori lipsă)
* **Tratarea outlierilor:** IQR / limitare percentile

### 4.2 Transformarea caracteristicilor

* **Normalizare:** Din rgb in bw si redimensionare la 128x128
* **Encoding pentru variabile categoriale**

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- ☑ Structură repository configurată
- ☐ Dataset analizat (EDA realizată)
- ☑ Date preprocesate
- ☑ Seturi train/val/test generate
- ☑ Documentație actualizată în README + `data/README.md`

---
