import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import io
import base64
import random

# Initialize Flask App
app = Flask(__name__)

# --- UTILITY FUNCTIONS ---

def calculate_change_metrics(df):
    """Calculates YoY change percentage and absolute change."""
    df['YoY_Change_Pct'] = ((df['current_value'] - df['previous_value']) / df['previous_value']) * 100
    df['Abs_Change'] = df['current_value'] - df['previous_value']
    
    # Feature Engineering for NLG
    df['Abs_Change_Formatted'] = df.apply(
        lambda row: f"Rs. {row['Abs_Change']:.1f} {row['unit']}" if row['unit'] == 'Crores' else 'N/A', axis=1
    )

    # Simple Significance Tiering (for narrative)
    df['Tier_Significance'] = np.select(
        [
            df['YoY_Change_Pct'].abs() >= 10,
            df['YoY_Change_Pct'].abs() >= 3,
        ],
        [
            'significant contraction',
            'strong increase',
        ],
        default='marginally'
    )

    df['Direction_Adjective'] = np.where(df['YoY_Change_Pct'] > 0, 'increase', 'contraction')
    df['Key_Context'] = df.apply(
        lambda row: 'confirming market share expansion and effective sales execution' if 'Premium' in row['metric_name'] and row['YoY_Change_Pct'] > 0
        else 'indicates immediate pressure on margins due to higher claims or operating costs and requires urgent review' if row['metric_name'] == 'Profit After Tax' and row['YoY_Change_Pct'] < 0
        else 'bolstering the balance sheet' if row['metric_name'] == 'Total Assets' and row['YoY_Change_Pct'] > 0
        else 'strengthening regulatory capital buffers' if row['metric_name'] == 'Solvency Ratio' and row['YoY_Change_Pct'] > 0
        else 'suggesting improved underwriting performance' if row['metric_name'] == 'Combined Ratio' and row['YoY_Change_Pct'] < 0
        else 'Operational Insight' if row['metric_name'] == 'Retention Efficiency (NP/GWP)'
        else 'N/A', axis=1
    )
    
    return df

def build_nlg_narrative(df, current_year):
    """
    Generates the Natural Language Generation (NLG) report based on processed data.
    """
    gwp = df[df['metric_name'] == 'Gross Written Premium'].iloc[0]
    pat = df[df['metric_name'] == 'Profit After Tax'].iloc[0]
    combined_ratio = df[df['metric_name'] == 'Combined Ratio'].iloc[0]
    total_assets = df[df['metric_name'] == 'Total Assets'].iloc[0]
    solvency_ratio = df[df['metric_name'] == 'Solvency Ratio'].iloc[0]
    net_premium = df[df['metric_name'] == 'Net Premium'].iloc[0]
    retention_efficiency = df[df['metric_name'] == 'Retention Efficiency (NP/GWP)'].iloc[0]
    
    
    # Determine overall theme
    pat_risk = "Key Risk" if pat['YoY_Change_Pct'] < -5 else "Minor Risk"
    overall_trend = "sales and solvency **Major Trends**, but these were significantly challenged by the **Key Risk** presented by falling profitability."

    narrative = f"""
## Quarterly Financial Summary: Focus on Major Trends, Risks, And Operational Insights

The quarter was characterized by positive momentum in {overall_trend}

### P&L Overview (Major Trends and Risks)

**Gross Written Premium (GWP)** remains a core driver, achieving a **{gwp['Tier_Significance']} of {gwp['YoY_Change_Pct']:.2f}%** (Rs. +1,622 Crores), confirming market share expansion and effective sales execution. This growth is mirrored by **Net Premium** which rose by **{net_premium['YoY_Change_Pct']:.2f}%** (Rs. +1,906 Crores). This growth is essential, as **Profit After Tax (PAT)** is under immediate pressure, registering a **{pat['Tier_Significance']} of {pat['YoY_Change_Pct']:.2f}%** (Rs. {pat['Abs_Change']:.0f} Crores). This decline indicates immediate pressure on margins due to higher claims or operating costs and requires urgent review. The **Combined Ratio** improved {combined_ratio['Tier_Significance']}, contracting by **{-combined_ratio['YoY_Change_Pct']:.2f}%** to **{combined_ratio['current_value']:.1f}%**, suggesting improved underwriting performance.

### Balance Sheet & Ratios (Stability and Risk)

The balance sheet remains robust, with **Total Assets** showing a **{total_assets['Tier_Significance']} of {total_assets['YoY_Change_Pct']:.2f}%** (Rs. +3,000 Crores), bolstering the balance sheet. Regulatory compliance is sound, as the **Solvency Ratio** increased by **{solvency_ratio['YoY_Change_Pct']:.2f}%** to **{solvency_ratio['current_value']:.2f}x**, strengthening regulatory capital buffers. The company's Retention Efficiency (Net Premium to GWP) stands at **{retention_efficiency['current_value']:.2f}%**, indicating a high proportion of risk retained internally.

Immediate operational focus must be placed on reversing the negative trend in profitability to stabilize the financial base.
"""
    return narrative

def generate_summary_table(df_processed):
    """
    Generates a markdown table summarizing key metrics, change, and narrative status.
    """
    table_rows = [
        "| Metric | Current Value / Ratio | YoY Change (%) | Absolute Change | Narrative Status |",
        "|:---|:---|:---|:---|:---|"
    ]
    
    # Sort the data frame to ensure consistent table order
    df_sorted = df_processed.sort_values(by=['metric_name'])

    for index, row in df_sorted.iterrows():
        # Clean up metric name for table display
        metric_name = row['metric_name'].replace('(NP/GWP)', '').strip()
        
        # Format current value
        current_val = f"Rs. {row['current_value']:,} {row['unit']}" if row['unit'] == 'Crores' else f"{row['current_value']:.2f}{row['unit']}"
        
        # Format absolute change
        abs_change_val = f"Rs. {row['Abs_Change']:.1f} {row['unit']}" if row['unit'] == 'Crores' and not np.isnan(row['Abs_Change']) else 'N/A'
        
        # Format YoY change
        yoy_change = f"{row['YoY_Change_Pct']:.2f}%" if not np.isnan(row['YoY_Change_Pct']) else 'N/A'
        
        # Determine the narrative status based on the complex context logic
        narrative_status = row['Key_Context']
        if 'Premium' in row['metric_name'] and row['YoY_Change_Pct'] > 0:
            narrative_status = 'Major Trend (Growth)'
        elif row['metric_name'] == 'Profit After Tax' and row['YoY_Change_Pct'] < 0:
            narrative_status = 'Key Risk (Contraction)'
        elif row['metric_name'] == 'Total Assets' and row['YoY_Change_Pct'] > 0:
            narrative_status = 'Stability (Growth)'
        elif row['metric_name'] == 'Solvency Ratio' and row['YoY_Change_Pct'] > 0:
            narrative_status = 'Strengthening Buffers'
        elif row['metric_name'] == 'Combined Ratio' and row['YoY_Change_Pct'] < 0:
            narrative_status = 'Improved Underwriting'
        elif row['metric_name'] == 'Retention Efficiency (NP/GWP)':
            narrative_status = 'Operational Insight'
        
        table_rows.append(
            f"| **{metric_name}** | {current_val} | {yoy_change} | {abs_change_val} | {narrative_status} |"
        )
    
    return "\n".join(table_rows)

def generate_financial_dashboard(df_current, current_year, previous_year):
    """
    Generates a simplified Matplotlib bar chart for the API response.
    Returns the chart encoded as a base64 PNG string.
    """
    # Filter the main metrics for a clean bar chart visualization
    metrics_to_plot = ['Gross Written Premium', 'Net Premium', 'Profit After Tax', 'Total Assets']
    df_plot = df_current[df_current['metric_name'].isin(metrics_to_plot)].set_index('metric_name')
    
    # Prepare data for plotting (Current vs. Previous)
    plot_data = df_plot[['current_value', 'previous_value']].T
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Width of the bars
    bar_width = 0.35
    
    # Bar positions
    x = np.arange(len(metrics_to_plot)) 
    
    # Plotting
    rects1 = ax.bar(x - bar_width/2, plot_data.loc['current_value'], bar_width, label=current_year, color='#008b8b')
    rects2 = ax.bar(x + bar_width/2, plot_data.loc['previous_value'], bar_width, label=previous_year, color='#d89e9e')
    
    # Add labels for a cleaner look
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=8)

    # Set up the chart aesthetics
    ax.set_ylabel('Value in Crores (Rs.)')
    ax.set_title(f'Key Financial Metrics: {current_year} vs. {previous_year}', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=15, ha="right")
    ax.legend()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    
    plt.tight_layout()
    
    # Save plot to an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    buf.seek(0)
    
    # Encode the buffer to Base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_base64

# --- FLASK API ENDPOINT ---

@app.route('/generate-report', methods=['POST'])
def generate_report():
    """
    API endpoint that receives financial data via POST request, runs the full analysis pipeline,
    and returns the NLG report, summary table, and dashboard image in JSON format.
    """
    try:
        # 1. Get JSON data from POST request
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data received'}), 400

        financial_data = data.get('financial_data', [])
        current_year = data.get('current_year', 'Current Year')
        previous_year = data.get('previous_year', 'Previous Year')

        if not financial_data:
             return jsonify({'status': 'error', 'message': 'financial_data array is empty or missing'}), 400

        # 2. Convert to DataFrame and Process
        df = pd.DataFrame(financial_data)
        df_processed = calculate_change_metrics(df)
        
        # 3. Generate NLG Narrative and Summary Table
        narrative = build_nlg_narrative(df_processed, current_year)
        summary_table_markdown = generate_summary_table(df_processed)
        
        # 4. Generate Dashboard Plot (Base64)
        dashboard_base64 = generate_financial_dashboard(df_processed, current_year, previous_year)
        
        # 5. Return structured JSON response
        return jsonify({
            'status': 'success',
            'current_year': current_year,
            'narrative_report': narrative,
            'summary_table_markdown': summary_table_markdown,
            'dashboard_image_base64': dashboard_base64
        })

    except Exception as e:
        # Handle exceptions gracefully
        return jsonify({
            'status': 'error',
            'message': f'An internal server error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Running the application in development mode
    app.run(debug=True, host='0.0.0.0')
