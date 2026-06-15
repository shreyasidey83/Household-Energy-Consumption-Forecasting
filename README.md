# 🏠 Household Energy Consumption Forecasting

> Predicting household electricity usage using **MLP** and **LSTM** deep learning models on 2+ million minute-level records.

---

## 📌 Overview

This project builds and compares two deep learning models for time-series forecasting of household power consumption using the [UCI Household Power Consumption Dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set).

| | |
|---|---|
| **Dataset** | 2,049,280 minute-level records (2006–2010) |
| **Target** | `Global_active_power` (kW) |
| **Models** | MLP vs LSTM |
| **Platform** | Kaggle (Tesla T4 GPU) |

---

## 📊 Results

| Model | RMSE (kW) | MAE (kW) | R² Score |
|-------|-----------|----------|----------|
| MLP   | 0.3356    | 0.1948   | 0.8494   |
| **LSTM** | **0.1911** | **0.0842** | **0.9512** |

✅ **LSTM wins** — 43% lower RMSE and explains 95.1% of variance in power consumption.

---

## 🗂️ Project Structure

```
├── Household_Energy_Consumption_Forecasting.ipynb   # Main notebook
└── README.md
```

---

## 🔧 Tech Stack

- **Python 3.12**
- **TensorFlow / Keras** — Model building & training
- **Scikit-learn** — Preprocessing & metrics
- **Pandas / NumPy** — Data manipulation
- **Matplotlib** — Visualizations

---

## 🚀 How to Run

### On Kaggle (Recommended)
1. Go to [Kaggle](https://www.kaggle.com/) and create a new notebook
2. Add the dataset: [Household Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)
3. Upload this notebook and run all cells
4. Enable **GPU accelerator** (Settings → Accelerator → GPU T4 x2)

### Locally
```bash
# Clone the repo
git clone https://github.com/shreyasidey83/Household-Energy-Consumption-Forecasting.git
cd Household-Energy-Consumption-Forecasting

# Install dependencies
pip install tensorflow pandas numpy scikit-learn matplotlib

# Download dataset from Kaggle and update the file path in Cell 2, then:
jupyter notebook Household_Energy_Consumption_Forecasting.ipynb
```

---

## 🧠 Model Architectures

### MLP
```
Input (60 timesteps × 7 features) → Flatten
→ Dense(128, ReLU) → Dropout(0.2)
→ Dense(64, ReLU)  → Dropout(0.2)
→ Dense(32, ReLU)  → Dense(1)
Total params: 64,257
```

### LSTM
```
Input (60 timesteps × 7 features)
→ LSTM(128, return_sequences=True) → Dropout(0.2)
→ LSTM(64)                         → Dropout(0.2)
→ Dense(32, ReLU) → Dense(1)
Total params: 121,153
```

---

## 📈 Key Visualizations

- Global Active Power time series (2006–2010)
- Distribution of power consumption
- MLP vs LSTM training loss curves
- Actual vs Predicted power (500 sample overlay)
- Scatter plots: Actual vs Predicted for both models
- Error distribution histograms

---

## 📦 Dataset

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) via Kaggle
- **Period**: December 2006 – November 2010
- **Frequency**: 1-minute intervals
- **Raw size**: 2,075,259 rows × 8 columns
- **After cleaning**: 2,049,280 rows (removed 25,979 missing rows)
- **Training subset**: Last 1,000,000 records

### Features Used
| Feature | Description |
|---------|-------------|
| `Global_active_power` | Household global active power (kW) — **target** |
| `Global_reactive_power` | Household global reactive power (kW) |
| `Voltage` | Minute-averaged voltage (V) |
| `Global_intensity` | Household global current intensity (A) |
| `Sub_metering_1` | Kitchen energy (Wh) |
| `Sub_metering_2` | Laundry room energy (Wh) |
| `Sub_metering_3` | Water heater & AC energy (Wh) |

---

## 🔮 Future Work

- [ ] GRU and Transformer-based architectures
- [ ] Add time-based features (hour, day of week, season)
- [ ] Multi-step forecasting (predict next N minutes)
- [ ] Hyperparameter tuning with Keras Tuner
- [ ] Deploy as a REST API

---

## 👤 Author

**Shreyasi Dey**  
[GitHub](https://github.com/shreyasidey83)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
