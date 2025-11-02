# 💡 Tips Prediction using Polynomial Regression

This project applies **Polynomial Regression** to predict the tip amount based on restaurant bills and group size.
It demonstrates how polynomial features can slightly improve prediction accuracy over simple linear regression.

---

## 📂 Project Structure

```
TipsPredict/
├── PolynomialRegression.py
├── tip.csv
├── requirements.txt
└── README.md
```

---

## 🧰 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/<your-username>/TipsPredict.git
cd TipsPredict
```

Make sure the dataset file `tip.csv` is in the same directory as the script.

Run the model:

```bash
python PolynomialRegression.py
```

---

## 🧮 Model Overview

* **Model A:** Uses more polynomial features (`total_bill²`, `size²`, and `total_bill × size`)
* **Model B:** Uses fewer features (`total_bill²` only)

Both models are trained using **Gradient Descent** to minimize **Mean Squared Error (MSE)**.

---

## 📈 Outputs

* Cost reduction over epochs plotted using Matplotlib
* Predicted tip values for new unseen data
* Comparison between simple and more complex models

**Example output:**

```
Predictions from Model A (More Features): [[2.94], [2.31], [2.33], [2.61], [3.27]]
Predictions from Model B (Fewer Features): [[2.97], [2.30], [2.32], [2.60], [3.31]]
```

---

## 🧩 Key Learnings

* Polynomial regression captures **nonlinear relationships** in data.
* However, beyond a certain degree, it may **overfit**.
* In this dataset, a second-degree polynomial offers only slight improvement — showing that the relationship is **almost linear**.
* Tips are more affected by **categorical and behavioral factors**, so predictions using only numeric features remain noisy.

---

## 🧑‍💻 Author

**Anish Ray**
Made with ❤️ using Python, NumPy, Pandas, and Matplotlib.

---

## 🪶 License

This project is open-source under the **MIT License**.

---
