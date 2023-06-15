import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define login credentials
LOGIN = 'username'
PASSWORD = 'password'

# Create login page
def login():
    # Get login credentials from user
    username = st.sidebar.text_input('Username')
    password = st.sidebar.text_input('Password', type='password')

    # Check credentials
    if username == LOGIN and password == PASSWORD:
        return True
    else:
        st.sidebar.error('Invalid username or password')
        return False

# Show login page
if login():
    st.title("Customer Segmentation App")

    # Upload CSV file
    csv_file = st.file_uploader("Upload CSV file", type=["csv"])

    # Load data into a pandas dataframe
    if csv_file is not None:
        df = pd.read_csv(csv_file)

        # Show raw data
        st.subheader("Raw Data")
        st.write(df)

        # Create features
        X = df.iloc[:, 2:].values

        # Elbow method to determine optimal number of clusters
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss)
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)

        # Cluster data
        n_clusters = st.slider("Number of clusters", 2, 10)
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_kmeans = kmeans.fit_predict(X)

        # Add cluster labels to dataframe
        df['Cluster'] = y_kmeans

        # Show clustered data
        st.subheader("Clustered Data")
        st.write(df)

        # Plot clusters
        fig, ax = plt.subplots()
        for i in range(n_clusters):
            ax.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, label=f'Cluster {i+1}')
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, marker='*', label='Centroids')
        ax.set_title('Clusters of customers')
        ax.set_xlabel('Average Return Rate')
        ax.set_ylabel('Total Spending')
        ax.legend()
        st.pyplot(fig)
