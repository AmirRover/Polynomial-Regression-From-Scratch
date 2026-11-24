# Nonlinear Regression: Sin(x) / Cos(x)

A Machine Learning project that approximates **Sine and Cosine functions** using Univariate Polynomial Regression, trained with Gradient Descent.

The model learns a polynomial approximation of the sine function from training data, evaluates its performance on unseen test data, and uses the trained weights to predict sine and cosine values for user-provided angles.

## Overview

This project demonstrates how polynomial feature expansion can be combined with linear regression to approximate nonlinear mathematical functions.

Instead of directly implementing the mathematical sine and cosine functions, the project trains a polynomial regression model to learn the underlying relationship from data.

**Key Highlights:**
- Univariate Polynomial Regression (Degree 5)
- Polynomial Feature Engineering
- Manual implementation of Gradient Descent
- Mean Squared Error (MSE) loss function
- R² Score for model evaluation
- Learning Curve Visualization
- Model Serialization using NumPy
- User-based prediction for Sine and Cosine

## Mathematical Background

### 1. Polynomial Regression

Polynomial Regression is an extension of Linear Regression that uses polynomial features to model nonlinear relationships.

The polynomial hypothesis function is defined as:

\[
\hat{y}=w_0+w_1x+w_2x^2+\cdots+w_dx^d
\]

Where:

- \(x\): Input feature
- \(d\): Polynomial degree
- \(w_i\): Model parameters (weights)
- \(\hat{y}\): Predicted output

In this project, a polynomial of degree 5 is used:

\[
\hat{y}=\sum_{i=0}^{5}w_ix^i
\]

### 2. Polynomial Feature Expansion

The input feature is transformed into a polynomial feature vector:

\[
\phi(x)=[1,x,x^2,x^3,x^4,x^5]
\]

For multiple samples, the prediction is calculated using matrix multiplication:

\[
\hat{Y}=\Phi(X)W
\]

### 3. Mean Squared Error (MSE)

The model uses Mean Squared Error to measure the difference between actual and predicted values.

\[
MSE=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
\]

Where \(m\) represents the number of training samples.

### 4. Gradient Descent

Gradient Descent is implemented manually to optimize the model weights by minimizing the MSE loss.

The gradient is calculated as:

\[
\nabla_W J(W)=\frac{2}{m}\Phi(X)^T(\hat{Y}-Y)
\]

The weights are updated using:

\[
W:=W-\eta\nabla_W J(W)
\]

Where:

- \(\eta\): Learning rate
- \(W\): Model weights
- \(\nabla_W J(W)\): Gradient of the loss function

### 5. Model Evaluation (R² Score)

The coefficient of determination (R²) is used to evaluate the model's performance on the test dataset.

\[
R^2=1-\frac{\sum_{i=1}^{m}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{m}(y_i-\bar{y})^2}
\]

A higher R² score indicates that the model explains a greater proportion of the variance in the target values.

## Project Structure

```text
Nonlinear Regression/
│
├── data/
│   ├── train-data.csv
│   └── test-data.csv
│
├── Nonlinear-Regression-Training-and-Evaluation.ipynb
│
├── Predict Model.ipynb
│
├── polynomial-linear-regression.npy
│
└── polynomial-linear-regression-univariate.npy
```

### File Descriptions

| File | Description |
|---|---|
| `data/train-data.csv` | Training dataset |
| `data/test-data.csv` | Testing dataset |
| `Nonlinear-Regression-Training-and-Evaluation.ipynb` | Model training, evaluation, and visualization |
| `Predict Model.ipynb` | User-based Sine and Cosine prediction |
| `polynomial-linear-regression.npy` | Saved trained model weights |
| `polynomial-linear-regression-univariate.npy` | Additional saved model parameters |

## Model Training & Evaluation

The training process is implemented in `Nonlinear-Regression-Training-and-Evaluation.ipynb`.

### Configuration

| Parameter | Value |
|---|---:|
| Regression Type | Univariate Polynomial Regression |
| Polynomial Degree | 5 |
| Learning Rate (η) | 0.6 |
| Epochs | 2000 |
| Optimization Algorithm | Gradient Descent |
| Loss Function | MSE |
| Evaluation Metric | R² Score |

### Training Pipeline

1. Initialize model weights randomly.
2. Load the training dataset.
3. Generate polynomial features.
4. Calculate predictions using Linear Regression.
5. Calculate MSE loss.
6. Compute gradients.
7. Update model weights using Gradient Descent.
8. Repeat the process for 2000 epochs.
9. Visualize the learning curve.
10. Save the trained weights using NumPy.

### Testing

After training, the model is evaluated using the independent test dataset.

The testing process includes:

- Loading the test dataset.
- Applying polynomial feature expansion.
- Generating predictions.
- Calculating the R² score.
- Visualizing training samples, testing samples, and the fitted polynomial curve.

### Model Visualization

The project visualizes the following:

- Training data points
- Testing data points
- Learned polynomial regression curve
- Training loss across epochs

## Sine & Cosine Prediction

The second notebook, `Predict Model.ipynb`, loads the saved model weights and allows users to predict Sine and Cosine values by entering an angle in degrees.

### Prediction Logic

The trained polynomial approximates the sine function using a normalized input:

\[
t=\frac{\theta}{180}
\]

The sine prediction is calculated as:

\[
\sin(\theta)\approx f\left(\frac{\theta}{180}\right)
\]

Cosine is calculated using the trigonometric identity:

\[
\cos(\theta)=
\begin{cases}
\sin(90^\circ-\theta), & 0\leq\theta\leq90^\circ\\
-\sin(\theta-90^\circ), & 90^\circ<\theta\leq180^\circ
\end{cases}
\]

This allows the same trained polynomial model to estimate both sine and cosine values without training a separate cosine model.

**Supported Input Range:** 0° to 180°.

### Example

```text
Enter your input: 30

Degree:  30.00 | Sin: 0.5000 | Cos: 0.8660
```

*Example output is illustrative; actual predictions depend on the trained model weights.*

## Installation & Requirements

### Prerequisites

- Python 3.12.10
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd "Nonlinear Regression"
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

## How to Run

**Step 1 — Train and Evaluate the Model**

Open and run:

```text
Nonlinear-Regression-Training-and-Evaluation.ipynb
```

Execute the notebook cells sequentially to train the polynomial regression model, evaluate its performance, and save the trained weights.

**Step 2 — Predict Sine and Cosine**

Open:

```text
Predict Model.ipynb
```

Run the notebook cells and enter an angle between 0 and 180 degrees when prompted.

The model will return the predicted Sine and Cosine values.

**Note:** Run the training notebook first if you need to generate a new `polynomial-linear-regression.npy` model file.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computations, matrix operations, and model serialization |
| Pandas | Dataset loading and data manipulation |
| Matplotlib | Data visualization and learning curves |
| Jupyter Notebook | Interactive development and experimentation |

## Author

**Amirreza Barati**

Machine Learning | Python | Mathematical Modeling

---

*This project is an educational implementation of Nonlinear Regression using Polynomial Feature Expansion and Gradient Descent.*
## A Final Note

Thank you for your time and support!

This project is just one implementation of Polynomial Regression. You can experiment with better datasets and different hyperparameters, such as polynomial degree, learning rate (η), and number of epochs, to potentially achieve even better results.

Feel free to experiment, improve the model, and see how far you can push its performance!

**Happy Learning & Happy Coding! 🚀**