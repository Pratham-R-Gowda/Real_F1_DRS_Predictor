# 🏁 Real F1 DRS Predictor: Surviving the Real-World Data Wall

## 👋 The Hook: It's Not Just Math, It's Physics

In Formula 1, predicting an overtake isn't as simple as asking, _"Who is going faster?"_ In a previous iteration of this project, I used synthetic (fake) data to prove how **SMOTE** (Synthetic Minority Over-sampling Technique) can fix the "Accuracy Paradox" of highly imbalanced data. But real-world F1 data is a completely different beast.

This project connects directly to AWS telemetry servers to pull 100% real racing data from the 2023 Brazilian Grand Prix. What I found was a perfect lesson in the limitations of algorithms and the absolute necessity of **Feature Engineering**.

---

## 🏎️ The Journey: From 5% to 33%

### 1. The "Junior" Model (The Accuracy Paradox)

Real F1 overtakes are incredibly rare. Out of all the valid racing laps in Brazil, only about **5%** resulted in an actual overtake.
When I fed raw Speed Trap and Tire Age data into a standard Random Forest, the model achieved a high overall accuracy by simply guessing "No Overtake" every time.

- **Result:** A terrible **5% Recall**. Out of all the real overtakes, the AI only spotted 1 of them. It was completely useless.

### 2. The SMOTE Fix (But Why Did It Stop at 29%?)

To fix the imbalance, I applied **SMOTE** to the training data to mathematically synthesize new overtakes, creating a 50/50 split.

- **Result:** Recall jumped to **29%**. A massive improvement, but still a failing grade for a production-ready Pit Wall tool.

**Why didn't SMOTE magically fix everything?**
Because SMOTE can fix _math_, but it cannot fix _missing physics_. In real life, a driver can have a massive speed advantage and fresh tires, but still fail to overtake because of:

1. **The DRS Train:** They are stuck behind 3 other cars who _also_ have DRS.
2. **Dirty Air:** They lose all their aerodynamic grip in the corners leading up to the straight, meaning they can't get close enough to make the move.

The AI didn't know about these things because I hadn't given it that data! SMOTE was trying its best with just Speed and Tire Age, but it hit a real-world wall.

### 3. The Feature Engineering Breakthrough (33% Recall)

To give the AI more context, I engineered a brand new feature: **`Pace_Advantage_Sec`**.

Instead of just looking at top speed on a straight, this feature calculates how much faster a driver is lapping the entire circuit compared to the field average. If a driver is fast on the straight _and_ has a terrible lap time, they are probably making mistakes in the corners and won't be close enough to overtake anyway.

- **The Final Result:** By providing the AI with this overall pace context, the model's Recall jumped to **33%**.
  While 33% might sound low to a layman, in the world of predicting highly chaotic, imbalanced physical events, engineering a feature that improves predictive power by over 13% is a massive win.

---

## 🛠️ The Tech Stack

- **Data Engineering:** `FastF1` (AWS Telemetry), `Pandas`, `NumPy`
- **Machine Learning:** `Scikit-Learn` (Random Forest)
- **Data Balancing:** `Imbalanced-Learn` (SMOTE)
- **Visualization/UI:** `Streamlit` (Custom HUD with dynamic strategy calls)

## 🚀 Run the Pit Wall Dashboard

Want to sit on the pit wall and calculate live overtake probabilities? Run this interactive dashboard locally.

**1. Clone this repository:**

```bash
git clone https://github.com/Pratham-R-Gowda/Real_F1_DRS_Predictor.git
cd Real-F1-DRS-Predictor
```
