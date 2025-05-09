from notebooks.eda_preprocessing import load_data
from src.rfm_engineering import compute_rfm
from src.rfm_engineering import assign_segments
from src.clustering import perform_clustering

def main():
    # Load cleaned dataset
    df = load_data("../output/cleaned_data.csv")

    # Compute RFM scores
    rfm_table = compute_rfm(df)

    # Assign segments based on RFM scores
    rfm_table["segments"] = rfm_table["RFM_score"].apply(lambda x: assign_segments(x))

    # Perform clustering
    clustered_data = perform_clustering(rfm_table)

    # Save the clustered data to a CSV file
    clustered_data.to_csv("clustered_data.csv", index=False)
    print("Clustered data saved to clustered_data.csv")
if __name__ == "__main__":
    main()
# This script is the main entry point for the RFM analysis and clustering process.