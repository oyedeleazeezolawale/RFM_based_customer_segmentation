import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import datetime as dt
import os
import warnings
warnings.filterwarnings("ignore")


#page config
st.set_page_config(
    page_title="BankTrust RFM Analysis Dashboard",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Load data with cache
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "../output/cleaned_data.csv")
    df = pd.read_csv(data_path)
    return df
df = load_data()

# Sidebar
with st.sidebar:
    st.image("assets/logo.png", width=200)
    st.title("BankTrust RFM Analysis Dashboard")
    st.markdown("This dashboard provides insights into customer behavior using RFM analysis.")
    st.markdown("### Select a date range:")
    start_date = st.date_input("Start date", value=df['date'].min(), min_value=df['date'].min(), max_value=df['date'].max())
    end_date = st.date_input("End date", value=df['date'].max(), min_value=df['date'].min(), max_value=df['date'].max())

    st.subheader("Select RFM Segments:")
    rfm_segments = st.multiselect(
        "Select RFM Segments",
        options=["Champions", "Loyal Customers", "Potential Loyalists", "New Customers", "Promising Customers", "At Risk", "About to Sleep", "Need Attention", "Lost"],
        default=["Champions", "Loyal Customers"]
    )

    st.subheader("Select Customer Segments:")
    customer_segments = st.multiselect(
        "Select Customer Segments",
        options=["High Value", "Medium Value", "Low Value"],
        default=["High Value"]
    )

st.subheader("Toggle View:")
show_cluster_distribution = st.checkbox("Show Cluster Distribution", value=True)
show_segment_funnel = st.checkbox("Show Segment Funnel", value=True)
show_rfm_matrix = st.checkbox("Show RFM Matrix", value=True)
show_segment_table = st.checkbox("Show Segment Table", value=True)
show_lifecycle_pie = st.checkbox("Show Lifecycle Pie Chart", value=True)
show_segment_composition = st.checkbox("Show Segment Composition Overview", value=True)
show_top_customers = st.checkbox("Show Top Customers", value=True)

st.markdown("---")
st.caption("Optimized for performance mobile and desktop devices.")

# Filter data based on user input
filtered_df = df[(df['segment'].isin(rfm_segments)) & (df['customer_segment'].isin(customer_segments)) & (df['date'] >= start_date) & (df['date'] <= end_date)]

#main title
st.title("BankTrust RFM Analysis Dashboard")
st.markdown("segment and priotize customers based on **recency**,**frequency**, and **Monetary** behavior for targeted marketing and retnetion strategies.")

#kpis
st.markdown("### Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(filtered_df['customer_id'].unique()))
col2.metric("average Recency", round(filtered_df['recency'].mean(), 2))
col3.metric("average Frequency", round(filtered_df['frequency'].mean(), 2))
col4.metric("average Monetary", round(filtered_df['monetary'].mean(), 2))

# === cluster Size Barchart ===
if show_cluster_distribution:
    st.markdown("### Cluster Size Distribution")
    cluster_count = filtered_df['cluster'].value_counts().sort_index_index()
    fig_cluster = px.bar(
        x=cluster_count.index,
        y=cluster_count.values,
        labels={'x': 'Cluster', 'y': 'Count'},
        title='Cluster Size Distribution',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

# === Segment Funnel ===
if show_segment_funnel:
    st.markdown("### segment value funnel")
    segment_monetary = filtered_df.groupby('segment')['monetary'].mean().sort_values(ascending=False).reset_index()
    fig_funnel = go.Figure(go.Funnel(
        y=segment_monetary['segment'],
        x=segment_monetary['monetary'],
        textinfo="value+percent initial",
        marker=dict(color='blue')
    ))
    fig_funnel.update_layout(yaxis_title="Segment", xaxis_title="Average Monetary")
    fig_funnel.update_layout(yaxis_title="Segment", xaxis_title="Average Monetary")
    st.plotly_chart(fig_funnel, use_container_width=True)  

# === Lifecycle Pie Chart ===
if show_lifecycle_pie:
    st.markdown("### 📊 Customer Lifecycle Distribution")
    seg_dist = filtered_df['Segment'].value_counts().reset_index()
    seg_dist.columns = ['Segment', 'Count']
    fig_pie = px.pie(seg_dist, names='Segment', values='Count', hole=0.4, title="Customer Lifecycle Segment Share")
    st.plotly_chart(fig_pie, use_container_width=True) 

# === Segment Composition Table ===
if show_segment_composition:
    st.markdown("### 📋 Segment Composition Overview")
    comp_df = filtered_df.groupby('Segment').agg({
        'CustomerID': 'count',
        'Monetary': 'sum'
    }).rename(columns={'CustomerID': 'Customers', 'Monetary': 'Total Monetary'}).reset_index()
    comp_df['% of Customers'] = (comp_df['Customers'] / comp_df['Customers'].sum() * 100).round(1)
    comp_df['% of Value'] = (comp_df['Total Monetary'] / comp_df['Total Monetary'].sum() * 100).round(1)
    fig_comp = px.bar(
    comp_df.sort_values(by='% of Value', ascending=False),
    x='Segment',
    y='% of Value',
    text='% of Value',
    color='Segment',
    title="Customer Value Contribution by Segment"
)
st.plotly_chart(fig_comp, use_container_width=True)

# === Top Customers Table ===
if show_top_customers:
    st.markdown("### 🏅 Top 10 High-Value Customers")
    top_customers = filtered_df.sort_values(by='Monetary', ascending=False).head(10)
    fig_top = px.bar(
    top_customers,
    x='CustomerID',
    y='Monetary',
    color='Segment',
    hover_data=['Recency', 'Frequency', 'Cluster'],
    title="Top 10 Customers by Monetary Value"
)
    st.plotly_chart(fig_top, use_container_width=True)
    st.dataframe(top_customers[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']], use_container_width=True)
    st.markdown("---")
    st.caption("Top customers based on monetary value.")

# === RFM Matrix ===
st.markdown("### 📊 RFM Matrix")
st.markdown("This matrix visualizes the distribution of customers across different RFM segments.")
cluster_size = filtered_df['segment'].value_counts().reset_index()
cluster_size.columns = ['Segment', 'Count']
fig = px.bar(cluster_size, x='Segment', y='Count', color='Segment', title="Cluster Size Distribution", text='Count')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(yaxis_title="Count", xaxis_title="Segment")
st.plotly_chart(fig, use_container_width=True)

# === Segment Table ===
if show_segment_table:
    st.markdown("### 📋 Segment Table")
    segment_table = filtered_df.groupby('segment').agg({
        'customer_id': 'nunique',
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean'
    }).reset_index()
    segment_table.columns = ['Segment', 'Customer Count', 'Average Recency', 'Average Frequency', 'Average Monetary']
    st.dataframe(segment_table, use_container_width=True)
    st.markdown("---")
    st.caption("Segment table showing average RFM values for each segment.")

# === RFM Distribution ===
st.markdown("### 📊 RFM Distribution")
st.markdown("This chart shows the distribution of customers across different RFM segments.")
fig_rfm = ff.create_distplot(
    [filtered_df['recency'], filtered_df['frequency'], filtered_df['monetary']],
    ['Recency', 'Frequency', 'Monetary'],
    show_hist=False,
    show_rug=True
)
fig_rfm.update_layout(title="RFM Distribution", xaxis_title="Value", yaxis_title="Density")
st.plotly_chart(fig_rfm, use_container_width=True)

# === RFM Heatmap ===
st.markdown("### 📊 RFM Heatmap")
st.markdown("This heatmap visualizes the correlation between RFM metrics.")
fig_heatmap = px.imshow(
    filtered_df[['recency', 'frequency', 'monetary']].corr(),
    color_continuous_scale='Viridis',
    title="RFM Heatmap",
    labels=dict(x="RFM Metrics", y="RFM Metrics")
)
fig_heatmap.update_xaxes(side="top")
st.plotly_chart(fig_heatmap, use_container_width=True)


# === RFM Segment Distribution ===
st.markdown("### 📊 RFM Segment Distribution")
st.markdown("This chart shows the distribution of customers across different RFM segments.")
fig_segment_dist = px.histogram(
    filtered_df,
    x='segment',
    color='segment',
    title="RFM Segment Distribution",
    labels={'segment': 'Segment'},
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig_segment_dist.update_layout(xaxis_title="Segment", yaxis_title="Count")
st.plotly_chart(fig_segment_dist, use_container_width=True)

# === RFM Score Distribution ===
st.markdown("### 📊 RFM Score Distribution")
st.markdown("This chart shows the distribution of customers based on their RFM scores.")
fig_rfm_score = px.histogram(
    filtered_df,
    x='RFM_score',
    color='segment',
    title="RFM Score Distribution",
    labels={'RFM_score': 'RFM Score'},
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig_rfm_score.update_layout(xaxis_title="RFM Score", yaxis_title="Count")
st.plotly_chart(fig_rfm_score, use_container_width=True)

# === RFM Score Boxplot ===
st.markdown("### 📊 RFM Score Boxplot")
st.markdown("This boxplot shows the distribution of RFM scores across different segments.")
fig_rfm_box = px.box(
    filtered_df,
    x='segment',
    y='RFM_score',
    color='segment',
    title="RFM Score Boxplot",
    labels={'segment': 'Segment', 'RFM_score': 'RFM Score'},
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig_rfm_box.update_layout(xaxis_title="Segment", yaxis_title="RFM Score")
st.plotly_chart(fig_rfm_box, use_container_width=True)

# === RFM Score Distribution by Segment ===
st.markdown("### 📊 RFM Score Distribution by Segment")
st.markdown("This chart shows the distribution of RFM scores across different segments.")
fig_rfm_score_segment = px.histogram(
    filtered_df,
    x='RFM_score',
    color='segment',
    title="RFM Score Distribution by Segment",
    labels={'RFM_score': 'RFM Score'},
    color_discrete_sequence=px.colors.qualitative.Plotly
)




# === Footer ===
st.markdown("---")
st.markdown("### About this Dashboard")
st.markdown("This dashboard provides insights into customer behavior using RFM analysis. It allows you to segment and prioritize customers based on recency, frequency, and monetary behavior for targeted marketing and retention strategies.")
st.markdown("### Contact")
st.markdown("For any inquiries or feedback, please contact us at [support@banktrust.com](mailto:support@banktrust.com)")

st.markdown("### Disclaimer")
st.markdown("This dashboard is for informational purposes only and should not be considered as financial advice. Please consult with a financial advisor for personalized recommendations.")
st.markdown("### License")          

st.markdown("This dashboard is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this dashboard as per the terms of the license.")
st.markdown("### Acknowledgements")
st.markdown("This dashboard was created using Streamlit, Plotly, and Pandas. Special thanks to the contributors of these libraries for their hard work and dedication.")
st.markdown("### Version")
st.markdown("1.0.0")
st.markdown("### Last Updated")
st.markdown(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# === End of Dashboard ===
st.markdown("---")
st.markdown("### Thank you for using the BankTrust RFM Analysis Dashboard!")

# === download button ===
st.markdown("### 💾 Export Data")
st.download_button("📥 Download Filtered Data as CSV", data=filtered_df.to_csv(index=False), file_name="filtered_rfm.csv")

