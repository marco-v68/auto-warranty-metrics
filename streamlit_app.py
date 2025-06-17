import streamlit as st
import pandas as pd
from datetime import datetime
import os
import plotly.express as px # For advanced visualizations
import plotly.graph_objects as go # For more control over heatmaps if needed
from typing import Optional # Required for type hinting in data_processing
from sklearn.preprocessing import MinMaxScaler # For heatmap data normalization

# Import your functions from the separate files
# Ensure these files (data_processing.py, safety_stock_logic.py) are in your GitHub repo
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

st.set_page_config(layout="wide", page_title="Inventory & Warranty Analytics", initial_sidebar_state="expanded")


# --- Custom Styling for a Cleaner Look ---
st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        background-color: #f0f2f6; /* Light gray background */
        color: #333333;
    }

    /* Streamlit's main content block - adjusting width and centering */
    .st-emotion-cache-1r6dm7m { /* This is a common class for the main content wrapper */
        padding-top: 2rem;
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 2rem;
        /* Remove explicit max-width here; rely on layout="wide" for initial width */
        /* You can adjust padding-left/right for more internal spacing if needed */
    }
    
    /* Ensure the main app content fills available width in wide mode */
    .st-emotion-cache-z5fcl4 { /* Sidebar width */
        width: 250px;
    }

    .st-emotion-cache-h4xjwx { /* Button styling */
        background-color: #007aff; /* Apple Blue */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: background-color 0.2s, box-shadow 0.2s;
    }
    .st-emotion-cache-h4xjwx:hover {
        background-color: #005bb5;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .st-emotion-cache-1v0mbdj { /* Metric container */
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 1.5rem;
    }
    
    .st-emotion-cache-1xv01z2 { /* Metric value */
        color: #007aff; /* Apple Blue for values */
        font-size: 2.2em;
        font-weight: 700;
    }
    
    .st-emotion-cache-czk5ad { /* Selectbox / Input field */
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
        padding: 0.5rem;
    }

    /* Expander styling */
    .st-emotion-cache-p2fwss { /* Expander header */
        border-radius: 8px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-weight: 600;
        color: #333333;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-p2fwss:hover {
        background-color: #f9f9f9;
    }
    .st-emotion-cache-p2fwss > div { /* Adjust padding inside expander header */
        padding: 0 !important; 
    }
    .st-emotion-cache-1aw8d9y { /* Expander content area */
        border: 1px solid #e0e0e0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        background-color: #ffffff;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* Tabs styling */
    .st-emotion-cache-l9bibm { /* Tab buttons container */
        background-color: #ffffff;
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-1cpx6a9 { /* Individual tab button */
        border-radius: 6px;
        font-weight: 500;
    }
    .st-emotion-cache-1cpx6a9.st-emotion-cache-1cpx6a9-hover { /* Hover state for tab */
        background-color: #e5e5ea;
    }
    .st-emotion-cache-1cpx6a9.st-emotion-cache-1cpx6a9-selected { /* Selected tab */
        background-color: #007aff !important;
        color: white !important;
    }
    .st-emotion-cache-1q1n064 { /* Tab content area */
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a;
        font-weight: 700;
    }
    h1 { font-size: 2.5em; }
    h2 { font-size: 2em; }
    h3 { font-size: 1.5em; }

</style>
""", unsafe_allow_html=True)

# Main container for the entire app content to ensure consistent padding and max-width
# Removed max-width from st-emotion-cache-1r6dm7m and adding a wrapper div in markdown
st.markdown('<div class="app-container">', unsafe_allow_html=True)

st.title("Comprehensive Inventory & Warranty Analytics")

# --- 1. Data Upload & Cleaning (Collapsible) ---
with st.expander("1. Data Upload & Cleaning", expanded=True):
    st.write("Upload your `warranty_raw.csv` file here to begin the analysis.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Please upload your raw warranty data in CSV format.")

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
            st.success("Data cleaned and ready for analysis! Preview below.")
            st.dataframe(st.session_state.cleaned_df.head(), use_container_width=True)
        else:
            st.error("Data cleaning resulted in an empty DataFrame. Please check your raw data and ensure it's correctly formatted.")
    else:
        st.info("No file uploaded yet. Please upload your `warranty_raw.csv` to proceed.")


# --- Only proceed if cleaned_df is available ---
if st.session_state.cleaned_df.empty:
    st.warning("Please upload a CSV file and ensure data is cleaned to proceed with analysis.")
    st.stop()


# --- 2. Analysis Parameters ---
st.header("2. Analysis Parameters")
as_of_date = st.date_input("Analysis 'As Of' Date", datetime.now(), help="Set the date for 'Year-to-Date' and 'Last 90 Days' calculations.")
as_of_datetime = datetime.combine(as_of_date, datetime.min.time())


# --- 3. Generate KPIs and Safety Stock ---
st.header("3. Generate KPI Reports & Safety Stock")

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

if 'generated_dfs' not in st.session_state:
    st.session_state.generated_dfs = {}

if st.button("Run All Analysis & Generate Reports", help="Click to generate all KPI tables and safety stock recommendations."):
    with st.spinner("Generating all KPI reports and Safety Stock data... This may take a few moments."):
        # Generate Safety Stock Table
        safety_stock_df = generate_safety_stock_table(
            cleaned_df=st.session_state.cleaned_df,
            safety_csv_path=list(output_paths.keys())[0],
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
        
    st.success("Analysis complete! Proceed to view results.")


    # --- Display Results & Reports (Only if data generated) ---
    if st.session_state.get('generated_dfs') and not st.session_state.generated_dfs['safety_stock_output.csv'].empty:
        st.header("4. Analysis Results & Reports")

        # --- Bloomberg Terminal Style - Key Metrics at the Top (for Safety Stock) ---
        st.subheader("Safety Stock Executive Summary")
        safety_stock_results_df = st.session_state.generated_dfs['safety_stock_output.csv']
        
        col1, col2, col3 = st.columns(3)
        total_ss_value = safety_stock_results_df['total_safety_stock_value'].sum()
        col1.metric("Total Safety Stock Value", f"${total_ss_value:,.2f}", help="Total calculated value of recommended safety stock across all items and locations.")

        critical_items_count = safety_stock_results_df[
            safety_stock_results_df['item_specific_z'] == 2.33
        ].shape[0]
        col2.metric("High Criticality Items (99% Service)", f"{critical_items_count}", help="Number of SKUs/locations with the highest business criticality (99% target service level).")

        avg_s_score = safety_stock_results_df['S_score'].mean()
        col3.metric("Average S-Score (Overall Risk)", f"{avg_s_score:.2f}", help="Average composite risk score, higher indicates higher overall risk for an item/location.")

        st.markdown("---") # Separator


        # --- Tabs for Detailed Reports and Visualizations ---
        tab_overview, tab_data, tab_visuals = st.tabs(["Overview & Key Visuals", "Detailed Reports", "Advanced Visualizations"])

        with tab_overview:
            st.subheader("Overview & Key Visuals")
            st.write("Quick insights and standard charts for safety stock data.")

            # Example: Total Safety Stock Value by Item Class (Bar Chart)
            st.write("#### Total Safety Stock Value by Item Class")
            ss_value_by_class = safety_stock_results_df.groupby('item_class')['total_safety_stock_value'].sum().sort_values(ascending=False)
            st.bar_chart(ss_value_by_class, use_container_width=True)

            # Example: Count of Items by Target Service Level (Z-score) (Bar Chart)
            st.write("#### Count of Items by Target Service Level (Z-score)")
            item_count_by_z = safety_stock_results_df['item_specific_z'].value_counts().sort_index()
            st.bar_chart(item_count_by_z, use_container_width=True)

        with tab_data:
            st.subheader("Detailed Reports")
            st.write("Browse and filter through all generated KPI tables.")

            report_options = list(output_paths.values())
            selected_report_display_name = st.selectbox(
                "Select a Report to View", 
                report_options, 
                help="Choose which detailed KPI report you'd like to examine."
            )

            selected_report_filename = next(key for key, value in output_paths.items() if value == selected_report_display_name)
            
            if selected_report_filename in st.session_state.generated_dfs:
                current_df_to_display = st.session_state.generated_dfs[selected_report_filename]

                if not current_df_to_display.empty:
                    st.write(f"#### {selected_report_display_name} Table")
                    
                    st.info("Use the filters below to refine your view of the table data.")
                    
                    # Filters for the table (Bloomberg style interaction)
                    filter_cols = ['item_sku', 'fulfillment_loc', 'item_class', 'customer_name', 'ship_state', 'month', 'customer_category', 'item_type']
                    available_filter_cols = [col for col in filter_cols if col in current_df_to_display.columns]

                    current_filtered_df = current_df_to_display.copy()

                    for col in available_filter_cols:
                        if col == 'item_sku' or col == 'customer_name': # Text input for search
                            search_term = st.text_input(f"Search by {col} (partial match)", key=f"filter_search_{selected_report_filename}_{col}", help=f"Enter text to filter {col}.")
                            if search_term:
                                current_filtered_df = current_filtered_df[
                                    current_filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
                                ]
                        else: # Selectbox for categories
                            all_unique_values = ['All'] + sorted(current_df_to_display[col].unique().tolist())
                            selected_value = st.selectbox(f"Filter by {col}", all_unique_values, key=f"filter_select_{selected_report_filename}_{col}", help=f"Select a specific {col} to filter.")
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
                    st.info(f"The selected report '{selected_report_display_name}' is empty. Please ensure data is available for this report type.")
            else:
                st.info("No report selected or report not generated yet. Please run the analysis first.")

        with tab_visuals:
            st.subheader("Advanced Safety Stock Visualizations")
            st.write("Explore interactive charts to gain deeper insights into safety stock data.")

            # 1. Scatter Plot: Risk Score vs. Total Safety Stock Value
            # Ensure data exists and relevant columns are present
            if not safety_stock_results_df.empty and \
               'S_score' in safety_stock_results_df.columns and \
               'total_safety_stock_value' in safety_stock_results_df.columns and \
               'item_class' in safety_stock_results_df.columns and \
               'item_sku' in safety_stock_results_df.columns and \
               'fulfillment_loc' in safety_stock_results_df.columns and \
               'safety_stock_qty' in safety_stock_results_df.columns and \
               'Reorder_Point' in safety_stock_results_df.columns:
                
                st.write("#### Risk Score vs. Total Safety Stock Value by Item Class")
                st.info("Hover over points for details. Adjust size based on Safety Stock Quantity.")
                
                # Ensure 'total_safety_stock_value' is numeric and handle potential NaNs before plotting
                plot_df_scatter = safety_stock_results_df.dropna(subset=['S_score', 'total_safety_stock_value', 'safety_stock_qty']).copy()
                
                if not plot_df_scatter.empty:
                    fig_scatter = px.scatter(
                        plot_df_scatter,
                        x="S_score",
                        y="total_safety_stock_value",
                        color="item_class", # Color by item class
                        hover_name="item_sku", # Show SKU on hover
                        hover_data=['fulfillment_loc', 'safety_stock_qty', 'Reorder_Point'], # Additional data on hover
                        size='safety_stock_qty', # Size bubbles by safety stock quantity
                        log_y=True, # Log scale for better visibility of cost differences
                        title="Safety Stock Risk (S_score) vs. Value (log scale)",
                        labels={
                            "S_score": "Composite Safety Score (Higher = More Risk)",
                            "total_safety_stock_value": "Total Safety Stock Value ($)",
                            "item_class": "Item Class"
                        }
                    )
                    fig_scatter.update_layout(height=500) # Set a fixed height for consistency
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("No data available for the Risk Score vs. Total Safety Stock Value scatter plot after cleaning.")
            else:
                st.info("Cannot generate Risk Score vs. Value scatter plot. Missing required columns or empty data after analysis.")


            # 2. Treemap: Safety Stock Value Breakdown
            # Ensure data exists and 'total_safety_stock_value' is present and > 0
            if not safety_stock_results_df.empty and \
               'total_safety_stock_value' in safety_stock_results_df.columns and \
               'fulfillment_loc' in safety_stock_results_df.columns and \
               'item_class' in safety_stock_results_df.columns and \
               'item_sku' in safety_stock_results_df.columns:
                
                st.write("#### Safety Stock Value Breakdown by Location and Item Class")
                st.info("Click on a box to drill down. Double-click to zoom out.")
                
                # Filter out rows where total_safety_stock_value is zero or NaN, as these break treemap
                plot_df_treemap = safety_stock_results_df[safety_stock_results_df['total_safety_stock_value'] > 0].copy()
                plot_df_treemap.loc[:, 'total_safety_stock_value'] = plot_df_treemap['total_safety_stock_value'].fillna(0)
                
                if not plot_df_treemap.empty:
                    fig_treemap = px.treemap(
                        plot_df_treemap,
                        path=[px.Constant("All Inventory"), 'fulfillment_loc', 'item_class', 'item_sku'],
                        values='total_safety_stock_value',
                        color='total_safety_stock_value', # Color intensity by value
                        hover_data=['safety_stock_qty', 'S_score'], # Add more hover info
                        title="Hierarchical View of Total Safety Stock Value Distribution"
                    )
                    fig_treemap.update_layout(margin = dict(t=50, l=25, r=25, b=25), height=600) # Adjust margins and set height
                    st.plotly_chart(fig_treemap, use_container_width=True)
                else:
                    st.warning("No data with positive 'Total Safety Stock Value' available for the Treemap after cleaning.")
            else:
                st.info("Cannot generate Safety Stock Value Treemap. Missing required columns or no positive safety stock values after analysis.")

            # 3. NEW: Interactive Heatmap for Service Center KPIs
            service_center_df = st.session_state.generated_dfs.get('service_center_kpi_output.csv')
            
            if service_center_df is not None and not service_center_df.empty:
                st.write("#### Service Center Performance Heatmap")
                st.info("This heatmap shows the relative intensity of various KPIs for each Service Center (Customer Name). Brighter colors indicate higher values after normalization.")

                # Select numerical columns for the heatmap
                heatmap_kpis = [
                    'total_service_center_cost',
                    'total_labor_cost',
                    'total_mileage_cost',
                    'avg_cost_per_visit',
                    'avg_units_serviced',
                    'bleed_per_unit',
                    'avg_margin_loss_per_visit',
                    'service_center_sku_diversity',
                    'service_center_customer_repeats'
                ]
                
                # Filter for only relevant KPI columns and drop rows with all NaN for selected KPIs
                # Also ensure customer_name is not NaN
                heatmap_df = service_center_df[['customer_name'] + [col for col in heatmap_kpis if col in service_center_df.columns]].copy()
                heatmap_df = heatmap_df.dropna(subset=[col for col in heatmap_kpis if col in heatmap_df.columns], how='all')
                heatmap_df = heatmap_df.dropna(subset=['customer_name'])

                if not heatmap_df.empty:
                    # Set customer_name as index for easier matrix formation
                    heatmap_df = heatmap_df.set_index('customer_name')
                    
                    # Handle infinities and large numbers, replace with NaN then impute or drop
                    # Also convert columns to numeric, coercing errors
                    for col in heatmap_df.columns:
                        if pd.api.types.is_numeric_dtype(heatmap_df[col]):
                            heatmap_df[col] = pd.to_numeric(heatmap_df[col], errors='coerce')
                            heatmap_df[col] = heatmap_df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Fill remaining NaNs with 0 for heatmap visualization (or mean/median if preferred)
                    heatmap_df = heatmap_df.fillna(0)

                    # Normalize the data for consistent color scaling across different KPIs
                    scaler = MinMaxScaler()
                    normalized_data = scaler.fit_transform(heatmap_df)
                    normalized_df = pd.DataFrame(normalized_data, columns=heatmap_df.columns, index=heatmap_df.index)

                    fig_heatmap = px.imshow(
                        normalized_df,
                        x=normalized_df.columns,
                        y=normalized_df.index,
                        color_continuous_scale=px.colors.sequential.Viridis, # A good sequential color scale
                        title="Normalized Service Center KPI Heatmap",
                        labels={
                            "x": "KPI",
                            "y": "Service Center (Customer Name)",
                            "color": "Normalized Value"
                        }
                    )
                    
                    fig_heatmap.update_xaxes(side="top") # Move KPI labels to top
                    fig_heatmap.update_layout(height=max(600, len(normalized_df.index) * 20), width=900) # Dynamic height based on number of customers
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.warning("No valid data available for the Service Center Performance Heatmap after cleaning and filtering.")
            else:
                st.info("Service Center KPI data not available or is empty. Run analysis first.")

# Close the main app-container div
st.markdown('</div>', unsafe_allow_html=True)
