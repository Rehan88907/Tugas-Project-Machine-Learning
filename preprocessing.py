import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)

    if 'Jenis_Produk' in df.columns:
        le = LabelEncoder()
        df['Jenis_Produk'] = le.fit_transform(df['Jenis_Produk'])

    X = df[['Jenis_Produk', 'Jumlah_Order', 'Harga']]
    y = df['Total']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
