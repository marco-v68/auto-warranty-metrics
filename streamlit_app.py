%%writefile streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import plotly.express as px # ADDED FOR ADVANCED VISUALIZATIONS

# Import your functions from the separate files
from data_processing import (
    load_and_clean,
    generate_sku_kpi_table,
    generate_customer_kpi_table,
    generate_location_kpi_table,
    generate_state_kpi_table,
    generate_monthly_kpi_table,
    generate_category_kpi_table,
    generate_item_class_kpi_table,
    generate_item_type_kpi_table,
    generate_service_center_kpi_table
)
from safety_stock_logic import generate_safety_stock_table

st.set_page_config(layout="wide")

st.title("Comprehensive Inventory & Warranty Analytics")

# --- Initial Data Upload and Cleaning ---
st.header("1. Data Upload & Cleaning")
st.write("Please upload your `warranty_raw.csv` file.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Use a session state variable to store cleaned_df after it's loaded
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = pd.DataFrame()

if uploaded_file is not None:
    # Save the uploaded file to a known path
    input_file_path = "uploaded_warranty_raw.csv"
    with open(input_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Load and clean the data
    with st.spinner("Cleaning and processing raw data..."):
        st.session_state.cleaned_df = load_and_clean(input_file_path)
    
    if not st.session_state.cleaned_df.empty:
        st.success("Data cleaned and ready for analysis!")
        st.write("Preview of cleaned data (first 5 rows):")
        st.dataframe(st.session_state.cleaned_df.head(), use_container_width=True)
    else:
        st.error("Data cleaning resulted in an empty DataFrame. Please check your raw data.")

# Only proceed if cleaned_df is available
if st.session_state.cleaned_df.empty:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop() # Stop execution until data is loaded

# --- Analysis Parameters ---
st.header("2. Analysis Parameters")
as_of_date = st.date_input("Analysis 'As Of' Date", datetime.now())
# Convert date_input to datetime object for functions
as_of_datetime = datetime.combine(as_of_date, datetime.min.time())

# --- Generate KPIs and Safety Stock ---
st.header("3. Generate KPI Reports & Safety Stock")

# Define output paths for all KPI tables
output_paths = {
    "safety_stock_output.csv": "Safety Stock Data",
    "sku_kpi_output.csv": "SKU KPIs",
    "customer_kpi_output.csv": "Customer KPIs",
    "location_kpi_output.csv": "Location KPIs",
    "state_kpi_output.csv": "State KPIs",
    "monthly_kpi_output.csv": "Monthly KPIs",
    "category_kpi_output.csv": "Customer Category KPIs",
    "item_class_kpi_output.csv": "Item Class KPIs",
    "item_type_kpi_output.csv": "Item Type KPIs",
    "service_center_kpi_output.csv": "Service Center KPIs"
}

# Use session state to store generated dataframes
if 'generated_dfs' not in st.session_state:
    st.session_state.generated_dfs = {}

if st.button("Run All Analysis & Generate Reports"):
    with st.spinner("Generating all KPI reports and Safety Stock data... This may take a few moments."):
        # Generate Safety Stock Table
        safety_stock_df = generate_safety_stock_table(
            cleaned_df=st.session_state.cleaned_df,
            safety_csv_path=list(output_paths.keys())[0], # Using the first key for safety stock
            as_of=as_of_datetime
        )
        st.session_state.generated_dfs['safety_stock_output.csv'] = safety_stock_df

        # Generate other KPI Tables
        st.session_state.generated_dfs['sku_kpi_output.csv'] = generate_sku_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[1], as_of=as_of_datetime)
        st.session_state.generated_dfs['customer_kpi_output.csv'] = generate_customer_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[2], as_of=as_of_datetime)
        st.session_state.generated_dfs['location_kpi_output.csv'] = generate_location_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[3], as_of=as_of_datetime)
        st.session_state.generated_dfs['state_kpi_output.csv'] = generate_state_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[4], as_of=as_of_datetime)
        st.session_state.generated_dfs['monthly_kpi_output.csv'] = generate_monthly_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[5], as_of=as_of_datetime)
        st.session_state.generated_dfs['category_kpi_output.csv'] = generate_category_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[6])
        st.session_state.generated_dfs['item_class_kpi_output.csv'] = generate_item_class_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[7])
        st.session_state.generated_dfs['item_type_kpi_output.csv'] = generate_item_type_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[8])
        st.session_state.generated_dfs['service_center_kpi_output.csv'] = generate_service_center_kpi_table(st.session_state.cleaned_df, list(output_paths.keys())[9])
        
    st.success("All analysis complete!")


# --- Display Results & Download Options ---
if st.session_state.get('generated_dfs') and not st.session_state.generated_dfs['safety_stock_output.csv'].empty:
    st.header("4. Analysis Results & Reports")

    # --- Bloomberg Terminal Style - Key Metrics at the Top (for Safety Stock) ---
    st.subheader("Safety Stock Executive Summary")
    safety_stock_results_df = st.session_state.generated_dfs['safety_stock_output.csv']
    col1, col2, col3 = st.columns(3)
    
    total_ss_value = safety_stock_results_df['total_safety_stock_value'].sum()
    col1.metric("Total Safety Stock Value", f"${total_ss_value:,.2f}")

    critical_items_count = safety_stock_results_df[
        safety_stock_results_df['item_specific_z'] == 2.33
    ].shape[0]
    col2.metric("High Criticality Items (99% Service)", f"{critical_items_count}")

    avg_s_score = safety_stock_results_df['S_score'].mean()
    col3.metric("Average S-Score (Overall Risk)", f"{avg_s_score:.2f}")

    st.markdown("---") # Separator

    # --- Select Report to View ---
    st.subheader("View Detailed Reports")
    report_options = list(output_paths.values())
    selected_report_display_name = st.selectbox("Select a Report to View", report_options)

    # Find the corresponding filename
    selected_report_filename = next(key for key, value in output_paths.items() if value == selected_report_display_name)
    
    if selected_report_filename in st.session_state.generated_dfs:
        current_df_to_display = st.session_state.generated_dfs[selected_report_filename]

        if not current_df_to_display.empty:
            st.write(f"### {selected_report_display_name} Table")
            
            # Filters for the table (Bloomberg style interaction)
            st.write("Apply filters to the current report:")
            
            # Dynamic filters based on available columns
            filter_cols = ['item_sku', 'fulfillment_loc', 'item_class', 'customer_name', 'ship_state', 'month', 'customer_category', 'item_type']
            available_filter_cols = [col for col in filter_cols if col in current_df_to_display.columns]

            current_filtered_df = current_df_to_display.copy()

            for col in available_filter_cols:
                if col == 'item_sku' or col == 'customer_name': # Text input for search
                    search_term = st.text_input(f"Search by {col} (partial match)", key=f"search_{col}")
                    if search_term:
                        current_filtered_df = current_filtered_df[
                            current_filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
                        ]
                else: # Selectbox for categories
                    all_unique_values = ['All'] + sorted(current_df_to_display[col].unique().tolist())
                    selected_value = st.selectbox(f"Filter by {col}", all_unique_values, key=f"filter_{col}")
                    if selected_value != 'All':
                        current_filtered_df = current_filtered_df[current_filtered_df[col] == selected_value]
            
            st.write(f"Displaying {len(current_filtered_df)} of {len(current_df_to_display)} entries.")
            st.dataframe(current_filtered_df, use_container_width=True)

            # --- Download Button for Current Report ---
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(current_df_to_display)
            st.download_button(
                label=f"Download {selected_report_display_name} as CSV",
                data=csv_data,
                file_name=selected_report_filename,
                mime="text/csv",
                key=f"download_{selected_report_filename}"
            )
        else:
            st.info(f"The selected report '{selected_report_display_name}' is empty.")
    else:
        st.info("No report selected or report not generated yet.")

    # --- Additional Visuals (Optional, but useful for Bloomberg style context) ---
    st.subheader("Key Visualizations for Safety Stock")
    
    # Example: Total Safety Stock Value by Item Class (Bar Chart)
    st.write("Total Safety Stock Value by Item Class:")
    ss_value_by_class = safety_stock_results_df.groupby('item_class')['total_safety_stock_value'].sum().sort_values(ascending=False)
    st.bar_chart(ss_value_by_class)

    # Example: Count of Items by Target Service Level (Z-score) (Bar Chart)
    st.write("Count of Items by Target Service Level (Z-score):")
    item_count_by_z = safety_stock_results_df['item_specific_z'].value_counts().sort_index()
    st.bar_chart(item_count_by_z)

    # --- NEW: Plotly Express Visualizations ---
    st.subheader("Interactive Safety Stock Visualizations")

    # 1. Scatter Plot: Risk Score vs. Total Safety Stock Value
    if not safety_stock_results_df.empty and \
       'S_score' in safety_stock_results_df.columns and \
       'total_safety_stock_value' in safety_stock_results_df.columns:
        
        st.write("#### Risk Score vs. Total Safety Stock Value by Item Class")
        fig_scatter = px.scatter(
            safety_stock_results_df,
            x="S_score",
            y="total_safety_stock_value",
            color="item_class", # Color by item class
            hover_name="item_sku", # Show SKU on hover
            hover_data=['fulfillment_loc', 'safety_stock_qty', 'Reorder_Point'], # Additional data on hover
            size='safety_stock_qty', # Size bubbles by safety stock quantity
            log_y=True, # Log scale for better visibility of cost differences
            title="Safety Stock Risk (S_score) vs. Value (log scale)"
        )
        fig_scatter.update_layout(xaxis_title="Composite Safety Score (Risk)",
                                  yaxis_title="Total Safety Stock Value (log scale)")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Cannot generate Risk Score vs. Value scatter plot. Missing required columns or empty data.")


    # 2. Treemap: Safety Stock Value Breakdown
    if not safety_stock_results_df.empty and 'total_safety_stock_value' in safety_stock_results_df.columns:
        st.write("#### Safety Stock Value Breakdown by Location and Item Class")
        fig_treemap = px.treemap(
            safety_stock_results_df,
            path=[px.Constant("All Locations"), 'fulfillment_loc', 'item_class', 'item_sku'],
            values='total_safety_stock_value',
            color='total_safety_stock_value', # Color intensity by value
            hover_data=['safety_stock_qty'],
            title="Hierarchical View of Total Safety Stock Value"
        )
        fig_treemap.update_layout(margin = dict(t=50, l=25, r=25, b=25)) # Adjust margins
        st.plotly_chart(fig_treemap, use_container_width=True)
    else:
        st.info("Cannot generate Safety Stock Value Treemap. Missing required columns or empty data.")


