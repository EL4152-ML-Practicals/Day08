# 🧠 Day 08 : Neural Networks & Classification

A comprehensive guide to implementing perceptrons and artificial neural networks for binary classification using Python, scikit-learn, and TensorFlow.

---

## 📚 Two Implementation Files

### 1️⃣ **Perceptron.ipynb** - Binary Perceptron Classifier

#### 🎯 What it does:

Implements a **basic perceptron from scratch** for binary classification using synthetic blobs dataset with decision boundary visualization.

#### 🔑 Key Steps:

```
📥 Generate Data: Binary classification blobs (150 samples, 2 features)
     ↓
✂️ Split Data: 80% train, 20% test
     ↓
🎨 Visualize: Scatter plot of training data
     ↓
🤖 Build Perceptron: Custom class with unit step activation
     ↓
🎓 Train Model: Iterative weight updates (1000 iterations)
     ↓
🎯 Make Predictions: Classify test samples
     ↓
📊 Evaluate: Calculate accuracy on test set
     ↓
📈 Visualize: Decision boundary line
```

#### 💡 Quick Memory:

- **Foundation of Neural Networks** 🏗️
- Hand-coded from scratch (no frameworks)
- Single hidden layer (implicit)
- Unit step activation function
- Shows how neurons learn decision boundaries

#### 🔢 Key Components:

```python
class Perceptron:
    - weights: Learn pattern strength for each feature
    - bias: Shift decision boundary
    - learning_rate: Controls update step size (0.01)
    - n_iters: Number of training iterations (1000)
```

#### 🔧 Training Algorithm:

```
For each iteration:
    For each training sample (x, y):
        1. Calculate linear output: x·w + b
        2. Apply unit step: if output ≥ 0 then 1, else 0
        3. Calculate error: (actual - predicted)
        4. Update weights: w = w + lr × error × x
        5. Update bias: b = b + lr × error
```

#### 📊 Activation Function:

$$f(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{if } x < 0 \end{cases}$$

#### 🔢 Key Variables:

- `X` → Feature data (n_samples × 2)
- `y` → Binary labels (0 or 1)
- `X_train, X_test` → Split features
- `p` → Perceptron instance
- `predictions` → Model output on test set
- `accuracy` → Classification accuracy (%)

---

### 2️⃣ **ANN.ipynb** - Artificial Neural Network (TensorFlow/Keras)

#### 🎯 What it does:

Implements a **deep neural network** for bank customer churn prediction using TensorFlow/Keras on real-world data with confusion matrix evaluation.

#### 🔑 Key Steps:

```
📥 Load Dataset: Churn_Modelling.csv (10,000 customers)
     ↓
🔍 Select Features: Extract columns 3 to -1 (skip ID, surname)
     ↓
🏷️ Encode Categorical Data:
   - Label encode 'Gender' (M/F → 0/1)
   - One-hot encode 'Geography' (France/Germany/Spain → 3 columns)
     ↓
📊 Prepare Data: Convert to numpy arrays
     ↓
✂️ Split Data: 80% train, 20% test
     ↓
📏 Feature Scaling: StandardScaler normalization
     ↓
🏗️ Build ANN Architecture:
   - Input Layer: (implicit, 12 features)
   - Hidden Layer 1: 6 neurons + ReLU activation
   - Hidden Layer 2: 6 neurons + ReLU activation
   - Output Layer: 1 neuron + Sigmoid activation
     ↓
⚙️ Compile Model: Adam optimizer + Binary Crossentropy
     ↓
🎓 Train: 100 epochs, batch size 32
     ↓
🔮 Make Predictions: Single customer & test set
     ↓
📊 Evaluate: Confusion matrix & accuracy score
```

#### 💡 Quick Memory:

- **Production-Ready Deep Learning** 🚀
- Uses TensorFlow/Keras (industry standard)
- Multiple hidden layers (deep)
- ReLU + Sigmoid activation functions
- Real customer dataset (10,000 samples)
- Complete ML pipeline with preprocessing

#### 🏗️ Network Architecture:

```
Input Features (12)
        ↓
   [Dense 6 neurons]
   ReLU activation
        ↓
   [Dense 6 neurons]
   ReLU activation
        ↓
   [Dense 1 neuron]
   Sigmoid activation
        ↓
   Output: Churn probability (0-1)
```

#### 🔧 Data Preprocessing Pipeline:

```python
1. Load CSV → DataFrame
2. Feature Selection → X[:, 3:-1], y[:, -1]
3. Label Encoding → Gender column
4. One-Hot Encoding → Geography column
5. Train/Test Split → 80/20
6. Feature Scaling → Standardize to mean=0, std=1
```

#### 📊 Activation Functions:

**Hidden Layers (ReLU):**
$$f(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

**Output Layer (Sigmoid):**
$$f(x) = \frac{1}{1 + e^{-x}}$$

#### 🔢 Key Variables:

- `df` → Loaded CSV dataset
- `x, y` → Features and target
- `x_train, x_test, y_train, y_test` → Split data
- `sc` → StandardScaler object
- `ann` → Sequential model instance
- `y_pred` → Model predictions on test set
- `cm` → Confusion matrix
- `accuracy` → Classification accuracy (%)

#### 📋 Model Compilation:

```python
ann.compile(
    optimizer='adam',          # Adaptive learning rate optimizer
    loss='binary_crossentropy', # For binary classification
    metrics=['accuracy']        # Monitor accuracy
)
```

#### 🎓 Training Configuration:

```python
ann.fit(
    x_train, y_train,
    batch_size=32,    # Process 32 samples per update
    epochs=100        # Train 100 times through dataset
)
```

#### 📊 Example Prediction:

```
Input: Customer from France, Credit Score 600, Male, 40 years old,
       3 years tenure, $60,000 balance, 2 products, Credit Card: Yes,
       Active Member: Yes, Salary: $50,000

Output: Probability > 0.5 → Customer will CHURN ❌
        Probability ≤ 0.5 → Customer will STAY ✅
```

---

## 🔄 Side-by-Side Comparison

```
┌──────────────────┬────────────────────┬──────────────────────┐
│ Feature          │ Perceptron.ipynb   │ ANN.ipynb            │
├──────────────────┼────────────────────┼──────────────────────┤
│ Library          │ NumPy (scratch)    │ TensorFlow/Keras     │
│ Dataset          │ Synthetic blobs    │ Real bank data       │
│ Samples          │ 150                │ 10,000               │
│ Features         │ 2                  │ 12                   │
│ Hidden Layers    │ 0 (single neuron)  │ 2 (6 + 6 neurons)    │
│ Activation Fn    │ Unit Step          │ ReLU + Sigmoid       │
│ Train/Test Split │ ✅ 80/20           │ ✅ 80/20             │
│ Preprocessing    │ ❌ None            │ ✅ Scaling + Encoding│
│ Data Type        │ 2D numeric         │ Mixed (categorical)  │
│ Visualization    │ ✅ Decision line   │ ❌ None (complex)    │
│ Metrics          │ Accuracy only      │ Confusion Matrix     │
│ Training Time    │ ⚡ Instant         │ ⏱️ Several seconds   │
│ Use Case         │ Learning basics    │ Production system    │
│ Complexity       │ ⭐ Beginner        │ ⭐⭐⭐ Advanced      │
└──────────────────┴────────────────────┴──────────────────────┘
```

---

## 🎓 Complete Machine Learning Workflow

### 🔄 Perceptron Workflow

```python
# 1. Import & Generate Data
import numpy as np
from sklearn.model_selection import train_test_split
X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2)

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# 3. Initialize & Train
p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)

# 4. Predict
predictions = p.predict(X_test)

# 5. Evaluate
accuracy = np.sum(y_test == predictions) / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 6. Visualize
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.plot([x0_1, x0_2], [x1_1, x1_2], 'k')  # Decision boundary
plt.show()
```

### 🔄 ANN Workflow

```python
# 1. Import & Load Data
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('Churn_Modelling.csv')

# 2. Preprocessing
# - Extract features & target
# - Encode categorical variables (Label + One-Hot)
# - Train/Test split
# - Feature scaling

# 3. Build Network
ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 4. Compile
ann.compile(optimizer='adam', loss='binary_crossentropy',
            metrics=['accuracy'])

# 5. Train
ann.fit(x_train, y_train, batch_size=32, epochs=100)

# 6. Predict
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)  # Convert probability to binary

# 7. Evaluate
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

---

## 📊 Performance Comparison

| Aspect                  | Perceptron   | ANN                     |
| ----------------------- | ------------ | ----------------------- |
| **Speed**               | ⚡ Very Fast | ⏱️ Moderate             |
| **Accuracy**            | 📊 ~80-90%   | 📊 ~85-95%              |
| **Complexity**          | 🎓 Simple    | 🚀 Complex              |
| **Interpretability**    | 🔍 Easy      | 🔍 Hard                 |
| **Real-world**          | ❌ Rarely    | ✅ Often                |
| **Scaling**             | ❌ Limited   | ✅ Excellent            |
| **Feature Engineering** | ❌ Manual    | ✅ Learns automatically |

---

## 🔑 Key Concepts

### Perceptron Concepts

1. **Weight Updates**: $w_i = w_i + \alpha \times (y - \hat{y}) \times x_i$
2. **Bias Update**: $b = b + \alpha \times (y - \hat{y})$
3. **Decision Boundary**: Linear line separating classes
4. **Convergence**: Reaches stable weights after iterations

### ANN Concepts

1. **Forward Propagation**: Input → Hidden layers → Output
2. **Backpropagation**: Calculate error gradients
3. **Gradient Descent**: Optimize weights iteratively
4. **Regularization**: Prevent overfitting
5. **Batch Processing**: Update weights after multiple samples

---

## 💡 When to Use What?

| Scenario                      | Use This                   |
| ----------------------------- | -------------------------- |
| 📚 Learning neural networks   | `Perceptron.ipynb`         |
| 🎓 Understanding fundamentals | `Perceptron.ipynb`         |
| 🏢 Production ML system       | `ANN.ipynb`                |
| 🔮 Complex data patterns      | `ANN.ipynb`                |
| 📊 Binary classification      | Both (choose by data size) |
| ⚡ Need fast predictions      | `Perceptron.ipynb`         |
| 🎯 Maximize accuracy          | `ANN.ipynb`                |
| 🔍 Interpretable model        | `Perceptron.ipynb`         |

---

## 📈 Typical Accuracy Results

```
Perceptron on synthetic data:
━━━━━━━━━━━━━━━━━━━━━━━━━ 85% ✓

ANN on bank churn data:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86% ✓
```

---

## 🔧 Prerequisites & Setup

```python
# Required Libraries
import pandas as pd          # Data manipulation
import numpy as np          # Numerical computing
import matplotlib.pyplot as plt  # Visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf     # Deep learning (ANN only)
```

### Installation (if needed):

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

---

## 🎯 Learning Path

```
Start Here
    ↓
1️⃣ Perceptron.ipynb
   ✅ Understand neurons
   ✅ Learn decision boundaries
   ✅ See training in action
    ↓
2️⃣ ANN.ipynb
   ✅ Build deeper networks
   ✅ Handle real data
   ✅ Apply preprocessing
   ✅ Use industry tools
    ↓
Advanced Topics
   → Convolutional Neural Networks (CNN)
   → Recurrent Neural Networks (RNN)
   → Hyperparameter tuning
   → Model deployment
```

---

## ✅ Key Takeaways

1. 🧠 **Neurons mimic brain learning** through weight adjustments
2. 🎓 **Perceptrons** are single-layer, good for learning
3. 🚀 **ANNs** are multi-layer, powerful for complex problems
4. 📊 **Preprocessing matters** (scaling, encoding, splitting)
5. 🎯 **Activation functions** add non-linearity and flexibility
6. 📈 **More layers ≠ Always better** (depends on data)
7. ⚙️ **Hyperparameters** significantly affect performance
8. 📋 **Evaluation metrics** help choose the best model

---

## 🚀 Quick Start

**For Perceptron**:

1. Open `Perceptron.ipynb`
2. Run cells top to bottom
3. Watch decision boundary form
4. Understand weight updates

**For ANN**:

1. Ensure `Churn_Modelling.csv` is in same directory
2. Open `ANN.ipynb`
3. Run preprocessing cells
4. Train the network
5. Check confusion matrix & accuracy

---

## 📚 Model Comparison Summary

```
Perceptron vs ANN

Input (2 features)
    ↓
Perceptron:        ANN:
[w1, w2]           [Dense 6]
   +b    (0 hl)       +ReLU
 sigmoid            [Dense 6]
   ↓                  +ReLU
Output             [Dense 1]
(binary)           +Sigmoid
                      ↓
                   Output
                   (binary)

Simple             Complex
Fast               Slower
80%+ accuracy      85%+ accuracy
```

---

**Created for EL 4152 - Machine Learning | Day 08 Practicals** 🎓

_Understanding both simple (Perceptron) and complex (ANN) approaches is crucial for mastering neural networks!_



























