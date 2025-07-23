import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import pickle
st.set_page_config(page_title="Customer segmentation app", layout="wide")
st.title("Customer segmentation using machine learning")
uploaded_file = st.sidebar.file_uploader("upload your csv data file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw data")
    st.dataframe(df)


    st.sidebar.subheader("Data cleaning options")
    auto_encode = st.sidebar.checkbox("Auto encode categorical columns", value=True)

    df_clean = df.copy()

    if auto_encode:
        for col in df_clean.select_dtypes(include=['object']).columns:
            if df_clean[col].nunique() <= 10:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            else:
                df_clean = df_clean.drop(columns=[col])  
    if "CUST_ID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["CUST_ID"])
    df_features = df_clean.select_dtypes(include=["int64", "float64"])
    df_features.fillna(df_features.mean(numeric_only=True), inplace=True)
    st.subheader("Correlation heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_features.corr(), cmap="coolwarm", annot=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Feature boxplot")
    selected_feature = st.selectbox("Select a feature", df_features.columns)
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df_features[selected_feature], ax=ax2)
    st.pyplot(fig2)

    st.sidebar.subheader("clustering options")
    algorithm = st.sidebar.selectbox("sellect clustering algorithm", ["Kmeans", "dbscan", "agglomerative"])

    if algorithm == "Kmeans":
        k = st.sidebar.slider("Number of clusters (K)", 2, 10, 4)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    elif algorithm == "dbscan":
        eps = st.sidebar.slider("dbscan: eps", 0.1, 5.0, 0.5)
        min_samples = st.sidebar.slider("dbscan: min_samples", 1, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        k = st.sidebar.slider("Number of clusters", 2, 10, 4)
        model = AgglomerativeClustering(n_clusters=k)




    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)

    
    clusters = model.fit_predict(scaled_features)
    df_clean["Cluster"] = clusters



    if algorithm == "KMeans":
        score = silhouette_score(scaled_features, clusters)
        st.success(f"Silhouette score: {score:.2f}")


    st.subheader("pca cluster visualization")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    df_clean["PCA1"] = pca_result[:, 0]
    df_clean["PCA2"] = pca_result[:, 1]

    fig3 = px.scatter(df_clean, x="PCA1", y="PCA2", color=df_clean["Cluster"].astype(str),
                      title="Clusters visualized with pca", hover_data=df_clean.columns)
    st.plotly_chart(fig3, use_container_width=True)

    
    if algorithm == "Kmeans":
        st.subheader("feaature importance")
        rf = RandomForestClassifier()
        rf.fit(df_features, df_clean["Cluster"])
        importances = pd.Series(rf.feature_importances_, index=df_features.columns)
        fig4 = px.bar(importances.sort_values(ascending=False), title="Feature importance")
        st.plotly_chart(fig4)

    
    if algorithm == "Kmeans":
        st.subheader("elbow Plot")
        distortions = []
        K_range = range(1, 11)
        for i in K_range:
            km = KMeans(n_clusters=i, random_state=42, n_init=10)
            km.fit(scaled_features)
            distortions.append(km.inertia_)

        fig5 = plt.figure()
        plt.plot(K_range, distortions, marker='o')
        plt.title("elbow method for optimal k")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        st.pyplot(fig5)

    st.subheader("Download clustered data")
    csv = df_clean.to_csv(index=False).encode()
    st.download_button("Download as csv", csv, "clustered_data.csv", "text/csv")


    if algorithm == "Kmeans":
        st.subheader("Download Kmeans model")
        model_pkl = pickle.dumps(model)
        st.download_button("Download Kmeans Model", model_pkl, "kmeans_model.pkl")

else:
    st.warning("Please upload a csv file to continue.")
