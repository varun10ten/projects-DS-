import json
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- 0. CONFIGURATION & MOCK DATA SETUP (Global Constants) ---

# Mock 3-year data for plotting (used directly in the visualization function)
THREE_YEAR_DATA = {
    'metric_name': ['Gross Written Premium', 'Net Premium', 'Net Earned Premium', 'Net Worth', 'Investment Income'],
    '2022-23': [38791, 31127, 30244, 19919, 10442],
    '2023-24': [41996, 34407, 34028, 21135, 9241],
    '2024-25': [43618, 36313, 33368, 21884, 8034]
}

# Define the custom color palette for the plots
COLOR_DUSTY_ROSE = '#d89e9e'   # GWP, Net Premium
COLOR_DARK_CYAN = '#008b8b'    # Net Earned Premium
COLOR_ORANGE = '#ff8c00'       # Net Worth (Line/Marker)
COLOR_PURPLE = '#9370db'       # Investment Income
COLOR_DARK_FILL = '#1e1e1e'    # Net Worth (Area Fill)

# --- 1. DATA READING (Simulated Database Schema and Join) ---

def mock_db_setup() -> pd.DataFrame:
    """
    Simulates the structure and content of the three financial tables (dim, fact_period) 
    and performs the necessary join to generate the final data frame with 6 metrics.
    """
    print("STEP 1: Data Reading and Schema Join Simulation...")
    
    # 1. dim_financial_table (The Dimension Table)
    dim_financial_table = pd.DataFrame({
        'metric_id': [1, 2, 3, 4, 5, 6],
        'metric_name': ['Gross Written Premium', 'Profit After Tax', 'Total Assets', 'Solvency Ratio (x)', 'Combined Ratio (%)', 'Net Premium'],
        'statement_type': ['P&L', 'P&L', 'Balance Sheet', 'Ratio', 'Ratio', 'P&L'],
        'category': ['Revenue', 'Profitability', 'Asset', 'Solvency', 'Profitability', 'Revenue'],
        'aggregation_logic': ['SUM', 'SUM', 'LAST_DAY_BALANCE', 'AVERAGE', 'AVERAGE', 'SUM'],
        'currency': ['INR', 'INR', 'INR', 'x', '%', 'INR']
    })

    # 2. fact_period_comparision (The Core Fact Table for Reporting)
    fact_period_comparision = pd.DataFrame({
        'metric_id': [1, 2, 3, 4, 5, 6],
        'reporting_date': ['2024-03-31'] * 6,
        'current_value': [43618.0, 3300.0, 118000.0, 1.75, 96.5, 36313.0],
        'prior_year_value': [41996.0, 3800.0, 115000.0, 1.68, 97.0, 34407.0],
        # Calculated YoY Change based on mock data:
        'yoy_change_per': [0.0386, -0.1316, 0.0261, 0.0417, -0.0052, 0.0554], 
        'mom_change_per': [0.005, 0.001, 0.002, 0.003, -0.001, 0.004], # Mock MoM changes
        'narrative_status': ['minor increase'] * 6
    })
    
    # Simulating the join
    report_facts = pd.merge(fact_period_comparision, dim_financial_table, on='metric_id', how='inner')
    
    # Convert reporting_date from object to datetime
    report_facts['reporting_date'] = pd.to_datetime(report_facts['reporting_date'])
    
    return report_facts

# --- 2. EDA (Exploratory Data Analysis) ---

def perform_eda(df: pd.DataFrame):
    """
    Simulates EDA by checking data types and unique categories.
    """
    print("\nSTEP 2: Exploratory Data Analysis (EDA) Simulation...")
    print("\n--- Data Types and Null Check (DataFrame Info) ---")
    df.info() 

# --- 3. FEATURE ENGINEERING ---

def feature_engineering_and_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Feature Engineering (Narrative Signals) based on the defined rules.
    """
    print("\nSTEP 3: Feature Engineering and OHE...")
    
    df['abs_change'] = df['current_value'] - df['prior_year_value']

    # 3a. Tier_Significance: Based on yoy_change_per thresholds
    def get_tier_significance(yoy_per):
        if abs(yoy_per) > 0.15:
            return "Tier 1 (Critical Change)"
        elif 0.05 <= abs(yoy_per) <= 0.15:
            return "Tier 2 (Notable Change)"
        else:
            return "Tier 3 (Minor/Stable)"
    df['Tier_Significance'] = df['yoy_change_per'].apply(get_tier_significance)
    
    # 3b. Direction_Adjective: High-impact vocabulary
    df['Direction_Adjective'] = df['yoy_change_per'].apply(lambda x: 
        'strong increase' if x > 0 else 'significant contraction')

    # 3c. Causal_Context: Based on metric_name and category
    def get_causal_context(row):
        metric = row['metric_name']
        is_increase = row['yoy_change_per'] > 0
        
        if metric == "Total Assets" and is_increase:
            return "bolstering the balance sheet and supporting future growth."
        elif metric == "Profit After Tax" and not is_increase:
            return "immediate pressure on margins due to higher claims or operating costs."
        elif metric == "Solvency Ratio (x)" and is_increase:
            return "strengthening regulatory capital buffers and stability."
        elif metric == "Combined Ratio (%)" and not is_increase: # Lower combined ratio is good
            return "improved underwriting performance and expense management."
        elif metric.endswith('Premium') and is_increase:
            return "market share expansion and effective sales execution."
        return "general operational performance." 
    
    df['Causal_Context'] = df.apply(get_causal_context, axis=1)

    # 3d. Format_Value_Cr: Custom formatting based on currency/type
    def format_value(row):
        value = row['current_value']
        currency = row['currency']
        
        if currency == 'INR':
            # Format as Rs. X,XXX Crores
            return f"Rs. {value:,.0f} Crores"
        elif currency == 'x':
            # Format as Ratio (e.g., 1.75x)
            return f"{value:.2f}x"
        elif currency == '%':
            # Format as Percentage (e.g., 96.5%)
            return f"{value:.1f}%"
        return str(value)

    df['Format_Value_Cr'] = df.apply(format_value, axis=1)

    df["Formatted_YoY_Per"] = (df["yoy_change_per"] * 100).round(2).astype(str) + '%'
    df["Formatted_Absolute_Change"] = df['abs_change'].map('{:+,}'.format)
    # Correctly append the currency and magnitude text
    df.loc[df['currency'] == 'INR', "Formatted_Absolute_Change"] = 'Rs. ' + df["Formatted_Absolute_Change"] + ' Crores'
    
    print("\nOHE Note: Categorical columns ('statement_type', 'category') would be OHE here for a traditional ML model.")
    
    return df

# --- 4. CALCULATING AND DEFINING KPIS ---

def calculate_and_define_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates and prints the core KPI values and returns key KPI results needed for the summary table.
    """
    print("\nSTEP 4: Calculating and Defining Project KPIs...")
    
    # Define metrics needed for KPI calculation
    gwp_row = df[df['metric_name'] == 'Gross Written Premium'].iloc[0]
    pat_row = df[df['metric_name'] == 'Profit After Tax'].iloc[0]
    np_row = df[df['metric_name'] == 'Net Premium'].iloc[0]
    asset_row = df[df['metric_name'] == 'Total Assets'].iloc[0]
    solvency_row = df[df['metric_name'] == 'Solvency Ratio (x)'].iloc[0]

    # Calculate KPI values
    retention_efficiency_value = ((np_row['current_value'] / gwp_row['current_value']) * 100)
    
    kpi_values = {
        "Premium Growth Rate (YoY)": f"{gwp_row['Formatted_YoY_Per']} ({gwp_row['Direction_Adjective']})",
        "Net Worth Stability (YoY)": f"{asset_row['Formatted_YoY_Per']} ({asset_row['Direction_Adjective']} in Total Assets)",
        "Retention Efficiency": f"{retention_efficiency_value:.2f}%", 
        "Profit Performance (YoY)": f"{pat_row['Formatted_YoY_Per']} ({pat_row['Direction_Adjective']})",
        "Current Solvency Ratio": solvency_row['Format_Value_Cr']
    }
    
    kpi_summary_df = pd.DataFrame(list(kpi_values.items()), columns=['KPI', 'Value'])
    
    print("\n--- CALCULATED KPI VALUES ---")
    print(kpi_summary_df.to_markdown(index=False))
    
    # Return key values needed for the final summary table
    return {"Retention Efficiency": retention_efficiency_value}

# --- 5. NLG MODEL BUILDING AND EXECUTION ---

def build_nlg_model(df: pd.DataFrame) -> str:
    """
    Constructs the prompt and executes the mock NLG generation call.
    """
    print("\nSTEP 5: NLG Model Building and Execution (Prompt Engineering)...")
    
    # Logic to identify if a metric needs urgent focus based on business logic
    def get_focus_area(row):
        metric = row['metric_name']
        yoy = row['yoy_change_per']
        
        if metric == 'Profit After Tax' and yoy < -0.10: # Significant drop in PAT
            return "Key Risk (Urgent Focus)"
        elif metric == 'Gross Written Premium' and yoy > 0.03:
            return "Major Trend (Growth)"
        elif row['Tier_Significance'] == "Tier 2 (Notable Change)":
            return "Focus Area (Watchlist)"
        return "Stable/Operational"

    df['Focus_Area'] = df.apply(get_focus_area, axis=1)

    # Simplified mock narrative generation based on the new metrics
    mock_narrative = f"""
## Quarterly Financial Summary: Focus on Major Trends, Risks, And Operational Insights

The quarter was characterized by positive momentum in sales and solvency **Major Trends**, but these were significantly challenged by the **Key Risk** presented by falling profitability.

### P&L Overview (Major Trends and Risks)

**Gross Written Premium (GWP)** remains a core driver, achieving a **strong increase of 3.86%** (Rs. +1,622 Crores), confirming market share expansion and effective sales execution. This growth is mirrored by **Net Premium** which rose by **5.54%** (Rs. +1,906 Crores). This growth is essential, as **Profit After Tax (PAT)** is under immediate pressure, registering a **significant contraction of 13.16%** (Rs. -500 Crores). This decline indicates immediate pressure on margins due to higher claims or operating costs and requires urgent review. The **Combined Ratio** improved marginally, contracting by **0.52%** to **96.5%**, suggesting improved underwriting performance.

### Balance Sheet & Ratios (Stability and Risk)

The balance sheet remains robust, with **Total Assets** showing a **strong increase of 2.61%** (Rs. +3,000 Crores), bolstering the balance sheet. Regulatory compliance is sound, as the **Solvency Ratio** increased by **4.17%** to **1.75x**, strengthening regulatory capital buffers. The company's Retention Efficiency (Net Premium to GWP) stands at **83.25%**, indicating a high proportion of risk retained internally.

Immediate operational focus must be placed on reversing the negative trend in profitability to stabilize the financial base.
"""
    
    # Print high-impact table
    high_impact_output = df[df['Focus_Area'].isin(["Major Trend (Growth)", "Key Risk (Urgent Focus)"])]
    
    high_impact_output = high_impact_output.copy()
    high_impact_output['MoM Change (%)'] = (high_impact_output['mom_change_per'] * 100).round(2)
    
    print("\n==================================================")
    print("      TABLE: HIGH-IMPACT METRICS & FOCUS AREAS")
    print("==================================================")
    print(high_impact_output[[
        'metric_name', 'Format_Value_Cr', 'Formatted_YoY_Per', 'MoM Change (%)', 'Focus_Area'
    ]].rename(columns={
        'metric_name': 'Metric Name', 
        'Format_Value_Cr': 'Current Value',
        'Formatted_YoY_Per': 'YoY Change (%)',
        'Focus_Area': 'Focus Area'
    }).to_markdown(index=False))
    print("==================================================")

    return mock_narrative

# --- 6. PLOTTING HELPER FUNCTIONS (for the Matplotlib Dashboard) ---

def format_y_axis(ax):
    """Formats the Y-axis to show 'Rs. in Crores' and thousand separators."""
    formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x):,}')
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylabel('Rs. in Crores', fontsize=10, fontweight='bold', color='gray')
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#e0e0e0')
    ax.set_facecolor('#f7f7f7')


def add_labels_to_bars(ax, bars):
    """Adds data labels above the bars."""
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval * 1.02, 
                f'{int(yval):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')


# --- 7. PLOTTING STYLIZED DASHBOARD ---

def generate_stylized_dashboard(df_plot: pd.DataFrame, years: List[str]):
    """
    Generates a 3x2 Matplotlib plot dashboard mimicking the requested style.
    """
    print("\nSTEP 7: Generating Graph for Prioritized Metrics (using matplotlib)...")

    # Create the figure and a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Flatten the 2D array of axes for easy iteration
    axes = axes.flatten()

    # --- 7a. Gross Written Premium (GWP) ---
    ax_gwp = axes[0]
    gwp_data = df_plot['Gross Written Premium']
    bars_gwp = ax_gwp.bar(years, gwp_data, color=COLOR_DUSTY_ROSE, alpha=0.8)
    ax_gwp.set_title('Gross Written Premium (Global)', fontsize=16, fontweight='bold', color='#1e3a8a')
    add_labels_to_bars(ax_gwp, bars_gwp)
    format_y_axis(ax_gwp)
    ax_gwp.set_ylim(bottom=gwp_data.min() * 0.9)

    # --- 7b. Net Premium ---
    ax_np = axes[1]
    np_data = df_plot['Net Premium']
    bars_np = ax_np.bar(years, np_data, color=COLOR_DUSTY_ROSE, alpha=0.8)
    ax_np.set_title('Net Premium (Global)', fontsize=16, fontweight='bold', color='#1e3a8a')
    add_labels_to_bars(ax_np, bars_np)
    format_y_axis(ax_np)
    ax_np.set_ylim(bottom=np_data.min() * 0.9)

    # --- 7c. Net Earned Premium ---
    ax_nep = axes[2]
    nep_data = df_plot['Net Earned Premium']
    bars_nep = ax_nep.bar(years, nep_data, color=COLOR_DARK_CYAN, alpha=0.9)
    ax_nep.set_title('Net Earned Premium (Global)', fontsize=16, fontweight='bold', color='#1e3a8a')
    add_labels_to_bars(ax_nep, bars_nep)
    format_y_axis(ax_nep)
    ax_nep.set_ylim(bottom=nep_data.min() * 0.9)

    # --- 7d. Net Worth (Stylized Area/Line Chart) ---
    ax_nw = axes[3]
    nw_data = df_plot['Net Worth']
    
    # 1. Fill the area (Black/Dark Fill)
    ax_nw.fill_between(years, nw_data, color=COLOR_DARK_FILL, alpha=1.0)
    
    # 2. Plot the line (Orange)
    ax_nw.plot(years, nw_data, color=COLOR_ORANGE, linewidth=4, zorder=3)
    
    # 3. Plot the markers (Orange Triangles)
    ax_nw.scatter(years, nw_data, color=COLOR_ORANGE, s=150, zorder=4, marker='^', 
                  edgecolor='black', linewidth=1.5)
    
    ax_nw.set_title('Net Worth', fontsize=16, fontweight='bold', color='#1e3a8a')
    
    # Add data labels aligned with the orange line/marker
    for i, (x, y) in enumerate(zip(years, nw_data.values)):
        ax_nw.text(x, y * 1.005, f'{int(y):,}', ha='center', fontsize=10, fontweight='bold', color=COLOR_ORANGE)
        
    format_y_axis(ax_nw)
    # Set tight Y-axis limits for visual impact
    ax_nw.set_ylim(bottom=19500, top=22500) 
    

    # --- 7e. Investment Income (Horizontal Bar Chart) ---
    ax_ii = axes[4]
    ii_data = df_plot['Investment Income'].sort_index(ascending=False)
    ii_years = ii_data.index.tolist()
    
    # Horizontal bars
    bars_ii = ax_ii.barh(ii_years, ii_data.values, color=COLOR_PURPLE, alpha=0.8)
    
    ax_ii.set_title('Investment Income', fontsize=16, fontweight='bold', color='#1e3a8a')
    ax_ii.set_xlabel('Rs. in Crores', fontsize=10, fontweight='bold', color='gray')
    ax_ii.set_ylabel('Financial Years', fontsize=10, fontweight='bold', color='gray')
    
    # Custom formatting for horizontal chart
    ax_ii.spines['top'].set_visible(False)
    ax_ii.spines['right'].set_visible(False)
    ax_ii.grid(axis='x', linestyle='--', alpha=0.5, color='#e0e0e0')
    ax_ii.set_facecolor('#f7f7f7')
    
    # Add data labels to the right of the bars
    for bar in bars_ii:
        width = bar.get_width()
        ax_ii.text(width + (ax_ii.get_xlim()[1] * 0.01), bar.get_y() + bar.get_height()/2, 
                   f'{int(width):,}', ha='left', va='center', fontsize=10, fontweight='bold')


    # --- 7f. Empty/Placeholder Subplot ---
    axes[5].set_visible(False) # Hide the last unused subplot

    # Set a Super Title for the whole dashboard
    fig.suptitle('Quarterly Financial Dashboard: Key Performance Trends (Rs. in Crores)', 
                 fontsize=22, fontweight='bold', color='#1e3a8a', y=0.95)

    plt.show() # Display the combined plot

# --- 8. FINAL TABLE GENERATION ---

def generate_final_summary_table(df: pd.DataFrame, retention_efficiency: float) -> str:
    """
    Generates the final summary table required by the user using the engineered DataFrame
    and the calculated Retention Efficiency.
    """
    
    # Select the core metrics for the table
    metrics_to_include = [
        'Gross Written Premium', 
        'Net Premium',
        'Profit After Tax', 
        'Combined Ratio (%)', 
        'Total Assets', 
        'Solvency Ratio (x)'
    ]
    
    # Filter the engineered data for the required metrics
    table_data = df[df['metric_name'].isin(metrics_to_include)].copy()

    # Map a simplified Narrative Status based on the original narrative
    def get_narrative_status(row):
        metric = row['metric_name']
        
        if metric == 'Profit After Tax':
            return "Key Risk (Contraction)"
        elif metric in ['Gross Written Premium', 'Net Premium']:
            return "Major Trend (Growth)"
        elif metric == 'Total Assets':
            return "Stability (Growth)"
        elif metric == 'Solvency Ratio (x)':
            return "Strengthening Buffers"
        elif metric == 'Combined Ratio (%)':
            return "Improved Underwriting"
        return "Operational Insight"
        
    table_data.loc[:, 'Narrative Status'] = table_data.apply(get_narrative_status, axis=1)
    
    # Create the Retention Efficiency row (Manually as it's a calculated KPI)
    retention_row = {
        'metric_name': 'Retention Efficiency (NP/GWP)',
        'Format_Value_Cr': f"{retention_efficiency:.2f}%",
        'Formatted_YoY_Per': 'N/A',
        'Formatted_Absolute_Change': 'N/A',
        'Narrative Status': 'Operational Insight'
    }
    
    # Use pd.concat for appending the new row
    table_data = pd.concat([table_data, pd.DataFrame([retention_row])], ignore_index=True)

    # Clean up the output dataframe for presentation
    final_table_df = table_data.rename(columns={
        'metric_name': 'Metric',
        'Format_Value_Cr': 'Current Value / Ratio',
        'Formatted_YoY_Per': 'YoY Change (%)',
        'Formatted_Absolute_Change': 'Absolute Change',
    })
    
    # Clean up Absolute Change for Ratios/Percentages where the unit is not "Crores" or is N/A
    final_table_df.loc[
        final_table_df['Metric'].isin(['Combined Ratio (%)', 'Solvency Ratio (x)', 'Retention Efficiency (NP/GWP)']), 
        'Absolute Change'
    ] = 'N/A'
    
    # Select and order final columns
    final_table_df = final_table_df[[
        'Metric', 
        'Current Value / Ratio', 
        'YoY Change (%)', 
        'Absolute Change', 
        'Narrative Status'
    ]]
    
    markdown_output = final_table_df.to_markdown(index=False)
    
    return markdown_output

# --- 9. MAIN PIPELINE EXECUTION FUNCTION ---

def main_nlg_pipeline():
    """Executes the full sequential pipeline."""
    print("STARTING FULL AUTOMATED NLG PIPELINE")
    print("==================================================")
    
    # --- 1-3: Data Processing and Feature Engineering ---
    df = mock_db_setup()
    print(f"1. Mock Data Loaded and 'reporting_date' formatted to {df['reporting_date'].dtype}.")
    print("--- Starting Data Processing Pipeline ---")
    perform_eda(df)
    df_engineered = feature_engineering_and_ohe(df)
    
    print("\n==================================================")
    print("       STEP 3: COMPLETE FEATURE ENGINEERED DATA")
    print("==================================================")
    engineered_cols = ['metric_name', 'current_value', 'yoy_change_per', 'Tier_Significance', 
                       'Direction_Adjective', 'Causal_Context', 'Format_Value_Cr', 
                       'Formatted_YoY_Per', 'Formatted_Absolute_Change']
    print(df_engineered[engineered_cols].to_markdown(index=False))
    print("==================================================")

    # --- 4-5: KPI Calculation and Narrative Generation ---
    kpi_results = calculate_and_define_kpis(df_engineered)
    generated_narrative = build_nlg_model(df_engineered)
    
    # --- 6: Plotting (Using the global THREE_YEAR_DATA) ---
    df_plot = pd.DataFrame(THREE_YEAR_DATA).set_index('metric_name').T
    years = df_plot.index.tolist()
    generate_stylized_dashboard(df_plot, years)
    
    # --- Final Output ---
    final_summary_table = generate_final_summary_table(df_engineered, kpi_results["Retention Efficiency"])
    
    print("==================================================")
    print("      FINAL NLG GENERATED REPORT")
    print("==================================================")
    print(generated_narrative)
    
    # Print the new summary table
    print("==================================================")
    print("      SUMMARY TABLE OF CORE FINANCIAL METRICS")
    print("==================================================")
    print(final_summary_table)
    print("==================================================")

    print("Pipeline Execution Complete.")

# Example execution block
if __name__ == "__main__":
    main_nlg_pipeline()
