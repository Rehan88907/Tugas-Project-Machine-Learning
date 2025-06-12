import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from train_model import train_and_save_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

history, model, X_test, y_test = train_and_save_model()

# Prediksi
y_pred = model.predict(X_test).flatten()

# Evaluasi
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Visualisasi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
plt.title('Loss Selama Training')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('MAE Selama Training')
plt.legend()

plt.tight_layout()
plt.show()

# Prediksi vs Aktual
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Aktual')
plt.ylabel('Prediksi')
plt.title('Prediksi vs Aktual Penjualan')
plt.grid(True)
plt.show()
