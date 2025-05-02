import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
# Charger les données et entraîner le modèle (à faire une seule fois)
data = pd.read_csv("Telco_Customer_Churn.csv")  # le chemin de votre fichier

# Convertir la colonne 'TotalCharges' en numérique, en remplaçant les erreurs par NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Maintenant, vous pouvez remplir les valeurs NaN avec la médiane
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Suppression de l'ID client qui n'est pas utile pour la modélisation
data.drop('customerID', axis=1, inplace=True)

# Encodage des variables catégorielles
le = LabelEncoder()
data['Churn'] = le.fit_transform(data['Churn'])  # No:0, Yes:1

# Encodage one-hot pour les variables catégorielles
data = pd.get_dummies(data, drop_first=True)

# Calcul de la matrice de corrélation
corr = data.corr()

# Filtrer les variables avec une corrélation supérieure à 0.2 avec 'churn'
correlation_threshold = 0.2
high_corr_features = corr[(corr['Churn'] > correlation_threshold) | (corr['Churn'] < -correlation_threshold)].index

# Créer un nouveau DataFrame avec ces variables
df1 = data[high_corr_features]

X = df1.drop("Churn", axis=1)
y = df1["Churn"]

# Créer un imputer (par exemple : remplacer par la médiane)
imputer = SimpleImputer(strategy='median')

# Appliquer à X
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Mise à l'échelle des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

# Sauvegarder le modèle et le scaler entraînés
with open('churn_model1.pkl', 'wb') as file:
    pickle.dump((model, scaler), file)


def predict_churn(input_data):
    # Chargement du modèle entraîné
    with open('churn_model1.pkl', 'rb') as file:
        model, scaler = pickle.load(file)

    # Préparer les données d'entrée pour la prédiction
    input_df = pd.DataFrame([input_data])

    # s'assurez que les colonnes de input_df correspondent à celles utilisées pour l'entraînement
    # et appliquez le même imputer et scaler
    input_df = pd.DataFrame(imputer.transform(input_df), columns=X.columns)
    input_df_scaled = scaler.transform(input_df)

    prediction = model.predict(input_df_scaled)
    probability = model.predict_proba(input_df_scaled)[0, 1]  # Probabilité de churn

    return prediction, probability


# Définir les entrées Streamlit
st.title("Prédiction de Churn Client")

tenure = st.slider("Durée de l'abonnement (en mois)", 0, 72, 12)  # Tenure est un entier, slider est approprié

internet_fiber = st.checkbox("Internet Fiber Optic")
internet_no = st.checkbox("Pas de service internet")

online_security_no_internet = st.checkbox("Online Security - Pas de service internet")
online_backup_no_internet = st.checkbox("Online Backup - Pas de service internet")
device_protection_no_internet = st.checkbox("Device Protection - Pas de service internet")
tech_support_no_internet = st.checkbox("Tech Support - Pas de service internet")
streaming_tv_no_internet = st.checkbox("Streaming TV - Pas de service internet")
streaming_movies_no_internet = st.checkbox("Streaming Movies - Pas de service internet")

contract_two_year = st.checkbox("Contrat de deux ans")
payment_electronic_check = st.checkbox("Paiement par chèque électronique")


# Créer le dictionnaire input_data avec les valeurs entrées par l'utilisateur
input_data = {
    'tenure': tenure,
    'InternetService_Fiber optic': int(internet_fiber),  # Convertir bool en int (0 ou 1)
    'InternetService_No': int(internet_no),
    'OnlineSecurity_No internet service': int(online_security_no_internet),
    'OnlineBackup_No internet service': int(online_backup_no_internet),
    'DeviceProtection_No internet service': int(device_protection_no_internet),
    'TechSupport_No internet service': int(tech_support_no_internet),
    'StreamingTV_No internet service': int(streaming_tv_no_internet),
    'StreamingMovies_No internet service': int(streaming_movies_no_internet),
    'Contract_Two year': int(contract_two_year),
    'PaymentMethod_Electronic check': int(payment_electronic_check)
}

# Bouton de prédiction
if st.button("Prédire le Churn"):
    prediction, probability = predict_churn(input_data)  # Appel de la fonction après l'entraînement

    if prediction[0] == 1:
        st.write(f"Le client risque de quitter (Churn) avec une probabilité de {probability:.2f}")
    else:
        st.write(f"Le client ne risque pas de quitter (No Churn) avec une probabilité de {1 - probability:.2f}")