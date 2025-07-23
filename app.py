import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the baseline trend data
df = pd.read_excel('test.xlsx', sheet_name='Sheet1')
# select 5th index row
row = df.iloc[5]

print(row)

# --- Helper Functions ---
def sample_from_distribution(dist_type, low=None, high=None, mean=None, std=None, left=None, mode=None, right=None):
    # Helper to check if all args are finite
    def all_finite(*args):
        for x in args:
            try:
                if x is None or not np.isfinite(float(x)):
                    return False
            except Exception:
                return False
        return True

    if dist_type == 'uniform':
        if not all_finite(low, high):
            return 0.0
        # Ensure low <= high
        if low > high:
            low, high = high, low
        if low == high:
            return low
        return np.random.uniform(low, high)
    elif dist_type == 'normal':
        if not all_finite(mean, std):
            return 0.0
        if std == 0:
            return mean
        return np.random.normal(mean, std)
    elif dist_type == 'triangular':
        if not all_finite(left, mode, right):
            return 0.0
        # Ensure left <= mode <= right
        vals = sorted([left, mode, right])
        left, mode, right = vals[0], vals[1], vals[2]
        if left == mode == right:
            return left
        return np.random.triangular(left, mode, right)
    elif dist_type == 'lognormal':
        if not all_finite(mean, std):
            return 0.0
        if std == 0:
            return np.exp(mean)
        return np.random.lognormal(mean, std)
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

def calculate_net_sales(final_baseline_trend, event_factors, class_share, product_share, gross_price, gtn):
    return final_baseline_trend * (1 + event_factors) * class_share * product_share * gross_price * (1 - gtn)

# --- Streamlit UI ---
st.title("Monte Carlo Simulations")

# --- Base Case from Excel Row ---
st.header("Base Case KPIs")
def clean_number(val):
    if isinstance(val, str):
        return round(float(val.replace('$','').replace(',','').strip()), 2)
    return round(float(val), 2)

base_final_baseline_trend = clean_number(row['Final Baseline Trend'])
base_event_factors = clean_number(row['Final Event Factor'])
base_class_share = clean_number(row['Class share'])
base_product_share = clean_number(row['Product Share'])
base_gross_price = clean_number(row['Gross Price SKU 1'])
base_gtn = clean_number(row['GTN for SKU 1'])

base_net_sales = calculate_net_sales(
    base_final_baseline_trend,
    base_event_factors,
    base_class_share,
    base_product_share,
    base_gross_price,
    base_gtn
)

st.markdown(f"**Base Case Net Sales:** `{base_net_sales:,.2f}`")
st.markdown(f"- Final Baseline Trend: `{base_final_baseline_trend:,.2f}`")
st.markdown(f"- Sum of Event Factors: `{base_event_factors:,.2f}`")
st.markdown(f"- ClassShare: `{base_class_share:,.2f}`")
st.markdown(f"- ProductShare: `{base_product_share:,.2f}`")
st.markdown(f"- GrossPrice: `{base_gross_price:,.2f}`")
st.markdown(f"- GTN: `{base_gtn:,.2f}`")

# User inputs
n_simulations = st.selectbox(
    "Number of Iterations (Simulations)",
    options=[1000, 10_000, 100_000, 1_000_000],
    index=1
)

# Toggle for event simulation level
event_simulation_mode = st.radio(
    "Event Simulation Mode:",
    options=["Individual Events", "Consolidated Events"],
    index=0,
    help="Individual: Simulate each of 15 events separately. Consolidated: Simulate all events as one combined factor."
)

st.header("Event Factors (Event1 - Event15)")

# Create a table-like structure with proper styling
st.markdown("""
<style>
.param-table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
}
.param-table th, .param-table td {
    border: 1px solid #ddd;
    text-align: center;
}
.param-table th {
    background-color: #f2f2f2;
    font-weight: bold;
}
.event-row {
    display: flex;
    align-items: center;
    border-bottom: 1px solid #eee;
}
.event-label {
    min-width: 80px;
    font-weight: bold;
    color: #666;
}

/* Style for input fields to make them distinct */
.stNumberInput > div > div > input {
    background-color: #e3f2fd !important
}

.stSelectbox > div > div > div {
    background-color: #e8f5e8 !important;
}
</style>
""", unsafe_allow_html=True)

# Table header
st.markdown("""
<table class="param-table">
<tr>
    <th>Event</th>
    <th>Distribution</th>
    <th>Downside</th>
    <th>Base Case</th>
    <th>Upside</th>
</tr>
</table>
""", unsafe_allow_html=True)

event_params = []
distribution_options = ['uniform', 'normal', 'triangular', 'lognormal']

if event_simulation_mode == "Individual Events":
    for i in range(1, 16):
        base_case_val = clean_number(row.get(f'Event{i}', 0.0))
        with st.container():
            cols = st.columns([1, 1.5, 1.5, 1, 1.5])
            with cols[0]:
                st.markdown(f"<div class='event-label'>Event {i}</div>", unsafe_allow_html=True)
            with cols[1]:
                dist_type = st.selectbox("Distribution", distribution_options, key=f"dist_{i}", label_visibility="collapsed")
            with cols[2]:
                downside = st.number_input("Downside", value=float(base_case_val), key=f"down_{i}", label_visibility="collapsed")
            with cols[3]:
                st.markdown(f"<div style='text-align:center; padding: 8px; background-color: #f8f9fa; border-radius: 4px;'>{base_case_val:.3f}</div>", unsafe_allow_html=True)
            with cols[4]:
                upside = st.number_input("Upside", value=float(base_case_val), key=f"up_{i}", label_visibility="collapsed")
        event_params.append({'downside': float(downside), 'upside': float(upside), 'dist_type': dist_type, 'base_case': base_case_val})

else:  # Consolidated Events
    st.markdown("**Consolidated Event Factor Parameters**")
    with st.container():
        cols = st.columns([1, 1.5, 1.5, 1, 1.5])
        with cols[0]:
            st.markdown(f"<div class='event-label'>All Events</div>", unsafe_allow_html=True)
        with cols[1]:
            consolidated_dist_type = st.selectbox("Distribution", distribution_options, key="consolidated_dist", label_visibility="collapsed")
        with cols[2]:
            consolidated_downside = st.number_input("Downside", value=float(base_event_factors), key="consolidated_down", label_visibility="collapsed")
        with cols[3]:
            st.markdown(f"<div style='text-align:center; padding: 8px; background-color: #f8f9fa; border-radius: 4px;'>{base_event_factors:.3f}</div>", unsafe_allow_html=True)
        with cols[4]:
            consolidated_upside = st.number_input("Upside", value=float(base_event_factors), key="consolidated_up", label_visibility="collapsed")
    
    # Store consolidated params in same format for compatibility
    consolidated_params = {'downside': float(consolidated_downside), 'upside': float(consolidated_upside), 'dist_type': consolidated_dist_type, 'base_case': base_event_factors}

st.header("Other Parameters")

# Table header for other parameters
st.markdown("""
<table class="param-table">
<tr>
    <th>Variable</th>
    <th>Distribution</th>
    <th>Downside</th>
    <th>Base Case</th>
    <th>Upside</th>
</tr>
</table>
""", unsafe_allow_html=True)

def styled_param_row(label, base_case_val, default_down, default_up, dist_key, down_key, up_key, step=0.01):
    dist_options = ['uniform', 'normal', 'triangular', 'lognormal']
    with st.container():
        cols = st.columns([1, 1.5, 1.5, 1, 1.5])
        with cols[0]:
            st.markdown(f"<div class='event-label'>{label}</div>", unsafe_allow_html=True)
        with cols[1]:
            dist_type = st.selectbox("Distribution", dist_options, key=dist_key, label_visibility="collapsed")
        with cols[2]:
            downside = st.number_input("Downside", value=float(base_case_val), key=down_key, label_visibility="collapsed", step=step)
        with cols[3]:
            st.markdown(f"<div style='text-align:center; padding: 10px; background-color: #f8f9fa; border-radius: 4px; margin: 5px 0;'>{base_case_val:.3f}</div>", unsafe_allow_html=True)
        with cols[4]:
            upside = st.number_input("Upside", value=float(base_case_val), key=up_key, label_visibility="collapsed", step=step)
    return {'downside': float(downside), 'upside': float(upside), 'dist_type': dist_type, 'base_case': base_case_val}

other_params = {}
other_params['ClassShare'] = styled_param_row(
    'ClassShare', base_class_share, base_class_share, base_class_share, 'dist_classshare', 'down_classshare', 'up_classshare')
other_params['ProductShare'] = styled_param_row(
    'ProductShare', base_product_share, base_product_share, base_product_share, 'dist_productshare', 'down_productshare', 'up_productshare')
other_params['GrossPrice'] = styled_param_row(
    'GrossPrice', base_gross_price, base_gross_price, base_gross_price, 'dist_grossprice', 'down_grossprice', 'up_grossprice', step=1.0)
other_params['GTN'] = styled_param_row(
    'GTN', base_gtn, base_gtn, base_gtn, 'dist_gtn', 'down_gtn', 'up_gtn')

if st.button("Run Simulation"):
    # Store individual event factors and other variables
    if event_simulation_mode == "Individual Events":
        individual_events = [[] for _ in range(15)]  # List for each of 15 events
    else:
        consolidated_events = []  # List for consolidated events
    
    class_share_list = []
    product_share_list = []
    gross_price_list = []
    gtn_list = []
    results = []

    for _ in range(int(n_simulations)):
        event_factors = 0
        
        if event_simulation_mode == "Individual Events":
            # Simulate each event individually
            for idx, event_param in enumerate(event_params):
                event_value = sample_from_distribution(
                    dist_type=event_param['dist_type'],
                    low=event_param['downside'], 
                    high=event_param['upside'],
                    mean=(event_param['downside'] + event_param['upside']) / 2,
                    std=(event_param['upside'] - event_param['downside']) / 6,
                    left=event_param['downside'], 
                    mode=(event_param['downside'] + event_param['upside']) / 2, 
                    right=event_param['upside']
                )
                individual_events[idx].append(event_value)  # Store individual event
                event_factors += event_value  # Sum for calculation
        else:
            # Simulate consolidated events
            event_factors = sample_from_distribution(
                dist_type=consolidated_params['dist_type'],
                low=consolidated_params['downside'], 
                high=consolidated_params['upside'],
                mean=(consolidated_params['downside'] + consolidated_params['upside']) / 2,
                std=(consolidated_params['upside'] - consolidated_params['downside']) / 6,
                left=consolidated_params['downside'], 
                mode=(consolidated_params['downside'] + consolidated_params['upside']) / 2, 
                right=consolidated_params['upside']
            )
            consolidated_events.append(event_factors)

        class_share = sample_from_distribution(
            dist_type=other_params['ClassShare']['dist_type'],
            low=other_params['ClassShare']['downside'], 
            high=other_params['ClassShare']['upside'],
            mean=(other_params['ClassShare']['downside'] + other_params['ClassShare']['upside']) / 2,
            std=(other_params['ClassShare']['upside'] - other_params['ClassShare']['downside']) / 6
        )
        class_share_list.append(class_share)

        product_share = sample_from_distribution(
            dist_type=other_params['ProductShare']['dist_type'],
            low=other_params['ProductShare']['downside'], 
            high=other_params['ProductShare']['upside'],
            mean=(other_params['ProductShare']['downside'] + other_params['ProductShare']['upside']) / 2,
            std=(other_params['ProductShare']['upside'] - other_params['ProductShare']['downside']) / 6
        )
        product_share_list.append(product_share)

        gross_price = sample_from_distribution(
            dist_type=other_params['GrossPrice']['dist_type'],
            low=other_params['GrossPrice']['downside'], 
            high=other_params['GrossPrice']['upside'],
            mean=(other_params['GrossPrice']['downside'] + other_params['GrossPrice']['upside']) / 2,
            std=(other_params['GrossPrice']['upside'] - other_params['GrossPrice']['downside']) / 6
        )
        gross_price_list.append(gross_price)

        gtn = sample_from_distribution(
            dist_type=other_params['GTN']['dist_type'],
            low=other_params['GTN']['downside'], 
            high=other_params['GTN']['upside'],
            mean=(other_params['GTN']['downside'] + other_params['GTN']['upside']) / 2,
            std=(other_params['GTN']['upside'] - other_params['GTN']['downside']) / 6
        )
        gtn_list.append(gtn)

        net_sales = calculate_net_sales(base_final_baseline_trend, event_factors, class_share, product_share, gross_price, gtn)
        results.append(net_sales)

    results = np.array(results)
    mean_sales = np.mean(results)
    median_sales = np.median(results)
    std_sales = np.std(results)
    p5 = np.percentile(results, 2.5)
    p95 = np.percentile(results, 97.5)

    st.subheader("Simulation Results")
    st.write(f"Mean Net Sales: {mean_sales:,.2f}")
    st.write(f"Median Net Sales: {median_sales:,.2f}")
    st.write(f"Std Dev: {std_sales:,.2f}")
    st.write(f"2.5th Percentile: {p5:,.2f}")
    st.write(f"97.5th Percentile: {p95:,.2f}")

    # Plot Net Sales Distribution
    st.subheader("Net Sales Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(results, kde=True, bins=50, color='skyblue', edgecolor='black', alpha=0.7, ax=ax)
    ax.axvline(mean_sales, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sales:,.0f}')
    ax.axvline(median_sales, color='green', linestyle='--', linewidth=2, label=f'Median: {median_sales:,.0f}')
    ax.axvline(p5, color='orange', linestyle=':', linewidth=2, label=f'2.5th Percentile: {p5:,.0f}')
    ax.axvline(p95, color='purple', linestyle=':', linewidth=2, label=f'97.5th Percentile: {p95:,.0f}')
    ax.axvline(base_net_sales, color='black', linestyle='-', linewidth=2, label=f'Base Case: {base_net_sales:,.0f}')
    ax.set_xlabel('Net Sales')
    ax.set_ylabel('Frequency')
    ax.set_title('Monte Carlo Simulation: Net Sales Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Plot Variable Distributions
    st.subheader("Variable Distributions")
    
    # Create subplots for other variables (4 main variables)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot ClassShare
    sns.histplot(class_share_list, kde=True, bins=30, ax=axes[0], color='lightgreen')
    axes[0].axvline(base_class_share, color='black', linestyle='--', linewidth=2, label=f'Base Case: {base_class_share:.3f}')
    axes[0].set_title('ClassShare Distribution')
    axes[0].set_xlabel('ClassShare')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot ProductShare
    sns.histplot(product_share_list, kde=True, bins=30, ax=axes[1], color='lightblue')
    axes[1].axvline(base_product_share, color='black', linestyle='--', linewidth=2, label=f'Base Case: {base_product_share:.3f}')
    axes[1].set_title('ProductShare Distribution')
    axes[1].set_xlabel('ProductShare')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot GrossPrice
    sns.histplot(gross_price_list, kde=True, bins=30, ax=axes[2], color='gold')
    axes[2].axvline(base_gross_price, color='black', linestyle='--', linewidth=2, label=f'Base Case: {base_gross_price:.0f}')
    axes[2].set_title('GrossPrice Distribution')
    axes[2].set_xlabel('GrossPrice')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot GTN
    sns.histplot(gtn_list, kde=True, bins=30, ax=axes[3], color='plum')
    axes[3].axvline(base_gtn, color='black', linestyle='--', linewidth=2, label=f'Base Case: {base_gtn:.3f}')
    axes[3].set_title('GTN Distribution')
    axes[3].set_xlabel('GTN')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Plot Individual Event Distributions (only for Individual Events mode)
    if event_simulation_mode == "Individual Events":
        st.subheader("Individual Event Factor Distributions")
        
        # Create a large subplot grid for all 15 events
        fig, axes = plt.subplots(5, 3, figsize=(18, 20))
        axes = axes.flatten()
        
        for i in range(15):
            if len(individual_events[i]) > 0:  # Only plot if there's data
                base_case_event = clean_number(row.get(f'Event{i+1}', 0.0))
                sns.histplot(individual_events[i], kde=True, bins=20, ax=axes[i], alpha=0.7)
                axes[i].axvline(base_case_event, color='red', linestyle='--', linewidth=2, label=f'Base: {base_case_event:.3f}')
                axes[i].set_title(f'Event {i+1} Distribution')
                axes[i].set_xlabel(f'Event {i+1} Factor')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f'Event {i+1}\nNo variation', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Event {i+1} (No variation)')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        # Plot Consolidated Event Distribution
        st.subheader("Consolidated Event Factor Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(consolidated_events, kde=True, bins=30, color='orange', alpha=0.7, ax=ax)
        ax.axvline(base_event_factors, color='red', linestyle='--', linewidth=2, label=f'Base: {base_event_factors:.3f}')
        ax.set_title('Consolidated Event Factor Distribution')
        ax.set_xlabel('Event Factor')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Summary Statistics Table
    st.subheader("Summary Statistics for Main Variables")
    summary_data = {
        'Variable': ['Net Sales', 'ClassShare', 'ProductShare', 'GrossPrice', 'GTN'],
        'Base Case': [base_net_sales, base_class_share, base_product_share, base_gross_price, base_gtn],
        'Mean': [mean_sales, np.mean(class_share_list), np.mean(product_share_list), np.mean(gross_price_list), np.mean(gtn_list)],
        'Std Dev': [std_sales, np.std(class_share_list), np.std(product_share_list), np.std(gross_price_list), np.std(gtn_list)],
        '2.5%': [p5, np.percentile(class_share_list, 2.5), np.percentile(product_share_list, 2.5), np.percentile(gross_price_list, 2.5), np.percentile(gtn_list, 2.5)],
        '97.5%': [p95, np.percentile(class_share_list, 97.5), np.percentile(product_share_list, 97.5), np.percentile(gross_price_list, 97.5), np.percentile(gtn_list, 97.5)]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(3)
    st.dataframe(summary_df, use_container_width=True)
    
    # Event Summary Table (conditional based on mode)
    if event_simulation_mode == "Individual Events":
        st.subheader("Individual Event Factors Summary")
        event_summary_data = {
            'Event': [f'Event {i+1}' for i in range(15)],
            'Base Case': [clean_number(row.get(f'Event{i+1}', 0.0)) for i in range(15)],
            'Mean': [np.mean(individual_events[i]) if len(individual_events[i]) > 0 else 0.0 for i in range(15)],
            'Std Dev': [np.std(individual_events[i]) if len(individual_events[i]) > 0 else 0.0 for i in range(15)],
            '2.5%': [np.percentile(individual_events[i], 2.5) if len(individual_events[i]) > 0 else 0.0 for i in range(15)],
            '97.5%': [np.percentile(individual_events[i], 97.5) if len(individual_events[i]) > 0 else 0.0 for i in range(15)]
        }
        
        event_summary_df = pd.DataFrame(event_summary_data)
        event_summary_df = event_summary_df.round(4)
        st.dataframe(event_summary_df, use_container_width=True)
    else:
        st.subheader("Consolidated Event Factor Summary")
        consolidated_summary_data = {
            'Event': ['Consolidated Events'],
            'Base Case': [base_event_factors],
            'Mean': [np.mean(consolidated_events)],
            'Std Dev': [np.std(consolidated_events)],
            '2.5%': [np.percentile(consolidated_events, 2.5)],
            '97.5%': [np.percentile(consolidated_events, 97.5)]
        }
        
        consolidated_summary_df = pd.DataFrame(consolidated_summary_data)
        consolidated_summary_df = consolidated_summary_df.round(4)
        st.dataframe(consolidated_summary_df, use_container_width=True)