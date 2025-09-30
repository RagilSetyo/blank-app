import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_percentage_error

# === Load Data ===
@st.cache_data
def load_data():
    return pd.read_csv("boston.csv")

boston = load_data()

# === Sidebar ===
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", [
    "Dataset", "EDA", "VIF & Korelasi", "Modeling", "Evaluasi dan Kesimpulan"
])
# Train Model
feature = boston.drop(columns="medv")
target = boston["medv"]

# First splitting: pretrain and test
feature_boston_pretrain, feature_boston_test, target_boston_pretrain, target_boston_test = train_test_split(feature, target, test_size=0.20, random_state=42)

# Second splitting: train and validation
feature_boston_train, feature_boston_validation, target_boston_train, target_boston_validation = train_test_split(feature_boston_pretrain, target_boston_pretrain, test_size=0.20, random_state=42)


X_boston_train = feature_boston_train.to_numpy()
y_boston_train = target_boston_train.to_numpy()
y_boston_train = y_boston_train.reshape(len(y_boston_train),)

X_boston_validation = feature_boston_validation.to_numpy()
y_boston_validation = target_boston_validation.to_numpy()
y_boston_validation = y_boston_validation.reshape(len(y_boston_validation),)

# === Persiapan Dataset ===
if menu == "Dataset":
    st.title("📊 Dataset Boston Housing")
    st.write(boston.head())

    st.subheader("Info Dataset")
    st.write("Jumlah baris:", len(boston))
    st.write("Jumlah kolom:", len(boston.columns))
    

    if st.checkbox("Tampilkan statistik deskriptif pada dataset boston"):
        st.write(boston.describe())

    if st.checkbox("Cek apakah ada missing value di data"):
        st.write(boston.isna().sum())

    if st.checkbox("Pengecekan duplikat pada data"):
        st.write("Hasil Duplikat:", len(boston) - len(boston.drop_duplicates()))

# === EDA ===
elif menu == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    if st.checkbox("Tampilkan distribusi target (medv)"):
        fig, ax = plt.subplots()
        sns.histplot(boston["medv"], kde=True, ax=ax)
        st.pyplot(fig)


# === Analisa korelasi & Seleksi fitur ===
elif menu == "VIF & Korelasi":
    st.title("📈 Analisa Korelasi & Seleksi Fitur")

    X = add_constant(feature_boston_train)

    vif_df = pd.DataFrame([vif(X.values, i)
               for i in range(X.shape[1])],
              index=X.columns).reset_index()
    vif_df.columns = ['feature','vif_score']
    vif_df = vif_df.loc[vif_df.feature!='const']

    st.subheader("VIF Score")
    st.write(vif_df)

    st.subheader("Heatmap Korelasi")
    boston_train = pd.concat([feature_boston_train, target_boston_train], axis=1)
    corr = boston.corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.write("""
    Terdapat kolom/fitur dengan VIF tinggi :
    - Tax = 8.268145
    - rad = 7.182410
    - nox = 4.585650
    - dis = 4.390458
    - indus = 3.741988""")

    # seleksi fitur, tidak menggunakan kolom rad, indus, dan nox
    feature_boston_train = feature_boston_train.drop(columns=['rad','nox', 'indus'])
    feature_boston_validation = feature_boston_validation.drop(columns=['rad','nox', 'indus'])
    feature_boston_test = feature_boston_test.drop(columns=['rad','nox', 'indus'])


    
    X = add_constant(feature_boston_train)

    vif_df = pd.DataFrame([vif(X.values, i)
               for i in range(X.shape[1])],
              index=X.columns).reset_index()
    vif_df.columns = ['feature','vif_score']
    vif_df = vif_df.loc[vif_df.feature!='const']
    st.subheader("VIF Score After")
    st.write(vif_df)

# === Modeling ===

elif menu == "Modeling":

    model_name = st.selectbox("Pilih Model", ["Ridge"])

    if model_name == "Ridge":
        alpha = st.selectbox("Alpha (Ridge)", [60.0])
        model = Ridge(alpha=alpha)
        st.session_state["alpha"] = alpha

    model.fit(X_boston_train, y_boston_train)
    y_pred = model.predict(X_boston_train)

    st.session_state["y_test"] = y_boston_train
    st.session_state["y_pred"] = y_pred

    st.success(f"Model {model_name} sudah dilatih!")


# === Evaluasi ===
elif menu == "Evaluasi dan Kesimpulan":
    st.title("📊 Evaluasi Model dan Kesimpulan")

    if "y_test" in st.session_state and "y_pred" in st.session_state:
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]

        # === Evaluasi model aktif ===
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        st.write(f"RMSE for training data is : {rmse:.2f}")
        st.write(f"MAPE for training data is : {mape:.2f}")
        st.write(f"MAE for training data is : {mae:.2f}")
        st.write(f"R2 for training data is : {r2:.2f}")


        # === Bandingkan beberapa alpha Ridge ===
        ridge_reg_40 = Ridge(alpha=40, random_state=42)
        ridge_reg_60 = Ridge(alpha=60, random_state=42)
        ridge_reg_80 = Ridge(alpha=80, random_state=42)
        ridge_reg_100 = Ridge(alpha=100, random_state=42)

        ridge_reg_40.fit(X_boston_train, y_boston_train)
        ridge_reg_60.fit(X_boston_train, y_boston_train)
        ridge_reg_80.fit(X_boston_train, y_boston_train)
        ridge_reg_100.fit(X_boston_train, y_boston_train)

        alphas = [40, 60, 80, 100]
        models = [ridge_reg_40, ridge_reg_60, ridge_reg_80, ridge_reg_100]

        results = []
        for model, alpha in zip(models, alphas):
            y_pred_val = model.predict(X_boston_validation)
            rmse_val = np.sqrt(mean_squared_error(y_boston_validation, y_pred_val))
            results.append({"Alpha": alpha, "RMSE Validation": rmse_val})

        st.subheader("Perbandingan RMSE pada Validation Set")
        st.dataframe(pd.DataFrame(results))

        # === Pilih model terbaik (misalnya alpha=60) ===
        ridge_best = ridge_reg_60
        coef_df = pd.DataFrame({
            "Feature": ["Intercept"] + feature_boston_train.columns.tolist(),
            "Coefficient": [ridge_best.intercept_] + list(ridge_best.coef_)
        })

        st.subheader("Koefisien Model Ridge (Ridge α=60)")
        st.dataframe(coef_df)

        st.write("""
                Penjelasan nilai Metrik model terbaik (Ridge) :
                Rata-rata prediksi meleset (MAE) = 3.52 * 1000 dollar per rumah
                Kesalahan Rata-rata (RMSE) = 4.88 * 1000 dollar per rumah
                Akurasi prediksi (MAPE) = 99.83%

                Total koefisien adalah 29.90, harga rumah di boston adalah 29.900 dollar. 
                Model memiliki prediksi meleset 3.520 - 4.880 dollar. 

                Artinya model akan memprediksi harga rumah di boston sebesar 29.900 dollar.
                Ada kasus dimana prediksi meleset sekitar sekitar 0.017 persen 
                sebesar 3.520 dan kadang 5080 dollar. 
                 
                 """)

    else:
        st.warning("Silakan latih model dulu di menu 'Modeling'.")



