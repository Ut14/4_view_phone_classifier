import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib


X = np.load("clip_features.npy")
y = np.load("clip_labels.npy")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

svm = SVC(probability=True)

param_grid = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto"]
}

print("🔍 Running Grid Search...")
grid = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy", verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"\n✅ Best Parameters: {grid.best_params_}")

y_pred = best_model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")


joblib.dump(best_model, "svm_phone_view_model.joblib")
print("💾 Best model saved as 'svm_phone_view_model.joblib'")
