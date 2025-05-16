import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from notebooks import eda_preprocessing
from src import Clustering



st.set_page_config(
    page_title="BankTrust RFM Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data ---
df = pd.read_csv("output/cleaned_data.csv")
if df.empty:
    st.warning("The dataset is empty. Please check the data source.")
else:
    st.success("Data loaded successfully!")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
# Display the logo
logo_path = os.path.join("asset/logo.png")
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, use_container_width = False)
else:
    st.warning("Logo not found. Please check the assets folder.")


page = st.sidebar.radio("📂 Navigation", ["🏠 Home", "📊 RFM engineering", "📈 Clustering", "👤 Customer Profile"])


if page == "🏠 Home":
    st.title("🏠 Welcome to the Customer Insights Dashboard ")
    st.markdown("""
        This Streamlit app dashboard provides insights into customer behaviour using both RFM and clustering analysis:

        - **RFM Segmentation**: Categorises customers based on their Recency, Frequency, and Monetary value.
        - **Clustering Analysis**: Uses machine learning to discover behavioural patterns in customers.

        Use the sidebar to navigate between analysis views.
    """)

    col1, col2, col3, col4 = st.columns(4)

    # Calculate metrics
    total_customers = df['CustomerID'].nunique() if 'CustomerID' in df.columns else len(df)
    avg_trxn = df['TransactionAmount (INR)'].mean() if 'TransactionAmount (INR)' in df.columns else 0
    avg_balance = df['CustAccountBalance'].mean() if 'CustAccountBalance' in df.columns else 0
    avg_age = df['age'].mean() if 'age' in df.columns else 0

    col1.metric("Total Customers", total_customers)
    col2.metric("Average Transaction Amount", f"₹{avg_trxn:,.0f}")
    col3.metric("Average Account Balance", f"₹{avg_balance:,.0f}")
    col4.metric("Average Age", f"{avg_age:.1f} yrs")

    # Display the first few rows of the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head(2))


     
    df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], dayfirst=True, errors='coerce')
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], dayfirst=True, errors='coerce')


    # Display the data types and null values
    st.subheader("Data Types and Null Values")
    st.write(df.dtypes)


    # Display the summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    st.markdown("""
    The summary statistics provide insights into the distribution of numerical features in the dataset.
 """)

    ## Age Distribution Chart using seaborn
    age_fig = px.histogram(
    df, 
    x='age', 
    nbins=20, 
    title='Customer Age Distribution',
    color_discrete_sequence=['indianred']
    )
    st.plotly_chart(age_fig, use_container_width=True)
    oc_df = (
        df['CustLocation']
        .value_counts()
        .nlargest(10)
        .reset_index()
    )
    oc_df.columns = ['CustLocation', 'Custcount']
    oc_fig = px.bar(
        oc_df,
        x='CustLocation',
        y='Custcount',
        color='CustLocation',
        color_continuous_scale='plasma',
        labels={'CustLocation': 'Location', 'Custcount': 'Customer Count'},
        title="Top 10 Customer Locations"
    )

    st.plotly_chart(oc_fig, use_container_width=True)
    
    
    #Create a pie chart for gender frequency
    gender_fig = px.pie(df, names='CustGender', title="Gender Distribution",
    color_discrete_sequence=['blue', 'pink'])
    st.plotly_chart(gender_fig, use_container_width=True)
    
    


elif page == "📊 RFM engineering":
    st.title("📊 RFM Engineering")
    st.write("This section will display RFM analysis and visualizations.")
    st.markdown("### RFM Segmentation")
    st.markdown("RFM (Recency, Frequency, Monetary) analysis is a marketing technique used to identify and segment customers based on their purchasing behavior. It helps businesses understand customer value and tailor marketing strategies accordingly.")

    # Removed unterminated st.markdown(""" line
    
    # --- Load Data ---
    df = pd.read_csv("output/rfm_segmented.csv")
    if df.empty:
       st.warning("The dataset is empty. Please check the data source.")
    else:
       st.success("Data loaded successfully!")

    # Define RFM segments based on unique values in the 'segments' column if it exists
    if 'segments' in df.columns:
        rfm_segments = df['segments'].unique().tolist()
    else:
        rfm_segments = []

    # Create a multiselect widget for RFM segments
    selected_segments = st.multiselect(
        "Select RFM Segments",
        options=rfm_segments,
        default=rfm_segments
    )

    def select_customer_segments():
        df['segments']   = df['segments'].value_counts()
        loyal_customers = df[df['segments'] == {'Loyal Customers'}.value_counts()]
        best_customers = df[df['segments'] == {'Best Customers'}.value_counts()]
        at_risk = df[df['segments'] == {'At Risk'}.value_counts()]
        churned = df[df['segments'] == {'Churned'}.value_counts()]
        
        st.markdown("### Select Customer Segments")
        st.markdown("Select the customer segments you want to analyze.")
        customer_segments = st.multiselect(
            "Select Customer Segments",
            options=["Best Customers", "Loyal Customers", "At Risk", "Churned"],
            default=["Best Customers"]
        )
        value_counts = df['Category'].value_counts()
        return value_counts

            
  
    st.subheader("Select Customer Segments:")
    customer_segments = st.multiselect(
        "Select Customer Segments",
        options=["Best Customers", "Loyal Customers", "At Risk", "Churned",],
        default=["Best Customers"]
    )
    

    
    st.markdown("### RFM Segmentation Overview")
    st.markdown("This section provides an overview of the RFM segmentation process and its results.")
    st.subheader("Toggle view")
    view_option = st.radio("", ["show_segment_funnel", "show_rfm_table"], index=0)
    
    # Define which visualizations to show based on the selected view_option
    show_rfm_table = view_option == "show_rfm_table"
    show_segment_funnel = view_option == "show_segment_funnel"
    
    # === RFM Distribution ===
    if show_rfm_table:
        st.markdown("### 📊 RFM Distribution")
        st.markdown("This table shows the distribution of customers across different RFM segments.")
        show_rfm_table = df.head(10)
        st.dataframe(show_rfm_table, use_container_width=True)


    # === Segment Funnel ===
    if show_segment_funnel:
        st.markdown("### segments value funnel")
        # Ensure filtered_df is defined before using it
        if 'segments' in df.columns and 'monetary' in df.columns:
            filtered_df = df.copy()
            segment_monetary = filtered_df.groupby('segments')['monetary'].mean().sort_values(ascending=False).reset_index()
            fig_funnel = go.Figure(go.Funnel(
                y=segment_monetary['segments'],
                x=segment_monetary['monetary'],
                textinfo="value+percent initial",
                marker=dict(color='blue')
            ))
            fig_funnel.update_layout(yaxis_title="Segments", xaxis_title="Average Monetary")
            fig_funnel.update_layout(yaxis_title="Segments", xaxis_title="Average Monetary")
            st.plotly_chart(fig_funnel, use_container_width=True)  
        else:
            st.warning("Required columns 'segments' or 'monetary' not found in the dataset.")

    st.markdown("---")
    st.caption("Optimized for performance mobile and desktop devices.")


elif page == "📈 Clustering":
    st.title("📈 Clustering Analysis")
    st.write("This section will display clustering analysis and visualizations.")
    st.markdown("### Clustering Analysis")
    st.markdown("Clustering analysis is a technique used to group similar data points together. In this case, we will use clustering to identify customer segments based on their purchasing behavior.")

    # --- Load Data ---
    df = pd.read_csv("output/clustered_data.csv")
    if df.empty:
        st.warning("The dataset is empty. Please check the data source.")
    else:
        st.success("Data loaded successfully!")

    # --- Clustering Analysis ---
    st.subheader("Clustering Analysis")
    st.markdown("This section provides an overview of the clustering analysis process and its results.")

    # Create a multiselect widget for clustering features
    clustering_features = df.columns.tolist()
    selected_features = st.multiselect(
        "Select Clustering Features",
        options=clustering_features,
        default=clustering_features
    )

    # Create a button to trigger clustering
    if st.button("Run Clustering"):
        if len(selected_features) < 2:
            st.warning("Please select at least two features for clustering.")
        else:
            # Perform clustering analysis
            clustered_df = Clustering.perform_clustering(df, selected_features)
            st.success("Clustering completed successfully!")

            # Display the clustered data
            st.subheader("Clustered Data")
            st.dataframe(clustered_df.head(10), use_container_width=True)

            # Plot the clusters
            fig_clusters = px.scatter(
                clustered_df,
                x=selected_features[0],
                y=selected_features[1],
                color='Cluster',
                title="Clusters Visualization",
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
    st.markdown("---")
    st.caption("Optimized for performance mobile and desktop devices.")
    st.markdown("### Clustering Analysis")
    st.markdown("This section provides an overview of the clustering analysis process and its results.")


elif page == "👤 Customer Profile":
    
    # --- Load Data ---
    df1 = pd.read_csv("output/cleaned_data.csv")
    df2 = pd.read_csv("output/rfm_segmented.csv")
    
    df = pd.merge(df1, df2[['CustomerID', 'Cluster']], on='CustomerID', how='left')
    if df.empty:
        st.warning("The dataset is empty. Please check the data source.")
    else:
        st.success("Data loaded successfully!")
    
    st.title("📊 Customer Profile Lookup")   
    customer_id_input = st.text_input("Enter Customer ID")
    st.write("For example C1010011")
    if customer_id_input:
        if customer_id_input in df["CustomerID"].astype(str).values:
            cust_profile = df[df["CustomerID"].astype(str) == customer_id_input].squeeze()
            st.subheader(f"Customer Profile: {cust_profile['CustomerID']}")
                        # Format Monetary with commas for readability
                #monetary_value = f"{cust_profile['Monetary']:,.0f}" if pd.notnull(cust_profile['Monetary']) else "N/A"

                # Format values into a table
            profile_data = {
                "Metric": ["Recency (days)", "Frequency", "Monetary (INR)", "Location", "Gender", "Age", "Account Balance", "Segment"],
                "Value": [
                        cust_profile.get("Recency", "N/A"),
                        cust_profile.get("Frequency", "N/A"),
                        cust_profile.get("Monetary","N/A"),
                        cust_profile.get("CustLocation", "N/A"),
                        cust_profile.get("CustGender", "N/A"),
                        cust_profile.get("CustomerAge", "N/A"), 
                        cust_profile.get("CustAccountBalance", "N/A"),
                        cust_profile.get("Segment", "N/A") 
                        ]
            }
            profile_df = pd.DataFrame(profile_data)

            st.table(profile_df)
            
        else:
            st.warning("Customer ID not found.")

    

    
