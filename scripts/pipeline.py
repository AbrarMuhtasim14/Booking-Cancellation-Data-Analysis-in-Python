"""
Data Pipeline, Model Training, and Figure Generation Script
Hospitality Booking Cancellation Prediction Project
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set high-DPI plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_preprocess_data():
    print("-> Loading raw datasets (2018, 2019, 2020)...")
    data_2018 = pd.read_csv(os.path.join(BASE_DIR, '2018.csv'))
    data_2019 = pd.read_csv(os.path.join(BASE_DIR, '2019.csv'))
    data_2020 = pd.read_csv(os.path.join(BASE_DIR, '2020.csv'))

    data_2018.columns = data_2018.columns.str.strip()
    data_2019.columns = data_2019.columns.str.strip()
    data_2020.columns = data_2020.columns.str.strip()

    hotel_data = pd.concat([data_2018, data_2019, data_2020], ignore_index=True)
    print(f"   Initial concatenated shape: {hotel_data.shape}")

    # Handling missing values
    hotel_data = hotel_data.dropna(subset=['children'])
    mode_country = hotel_data['country'].mode()[0]
    hotel_data['country'] = hotel_data['country'].fillna(mode_country)
    if 'company' in hotel_data.columns:
        hotel_data = hotel_data.drop(columns=['company'])
    hotel_data['agent'] = hotel_data['agent'].fillna(9)

    # Load and merge auxiliary tables
    market_segment_data = pd.read_csv(os.path.join(BASE_DIR, 'market_segment.csv'))
    meal_cost_data = pd.read_csv(os.path.join(BASE_DIR, 'meal_cost.csv'))

    hotel_data['market_segment'] = hotel_data['market_segment'].astype(str).str.strip()
    market_segment_data['market_segment'] = market_segment_data['market_segment'].astype(str).str.strip()

    meal_cost_data['meal'] = meal_cost_data['meal'].astype(str).str.strip()
    meal_cost_data['Cost'] = pd.to_numeric(meal_cost_data['Cost'].astype(str).str.strip(), errors='coerce')
    hotel_data['meal'] = hotel_data['meal'].astype(str).str.strip()

    hotel_data = pd.merge(hotel_data, market_segment_data, on='market_segment', how='left')
    hotel_data = pd.merge(hotel_data, meal_cost_data, on='meal', how='left')

    # Date anomaly filtering
    hotel_data['reservation_status_date'] = pd.to_datetime(hotel_data['reservation_status_date'])
    expected_years = [2018, 2019, 2020]
    hotel_data = hotel_data[hotel_data['reservation_status_date'].dt.year.isin(expected_years)]

    # Deduplication
    dups_count = hotel_data.duplicated().sum()
    print(f"   Deduplicating: Removing {dups_count} duplicate rows...")
    hotel_data = hotel_data.drop_duplicates().reset_index(drop=True)
    print(f"   Final clean records: {len(hotel_data)} rows across {hotel_data.shape[1]} columns.")

    return hotel_data


def generate_eda_figures(hotel_data):
    print("-> Generating high-resolution publication figures in assets/...")

    # 1. Monthly Cancellation Trend
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_map = {m: abbr for m, abbr in zip(month_order, month_abbr)}

    monthly_rates = hotel_data.groupby(['arrival_date_year', 'arrival_date_month'])['is_canceled'].mean().reset_index()
    monthly_rates['month_idx'] = monthly_rates['arrival_date_month'].map(lambda x: month_order.index(x) if x in month_order else -1)
    monthly_rates = monthly_rates.sort_values(by=['arrival_date_year', 'month_idx'])
    monthly_rates['month_abbr'] = monthly_rates['arrival_date_month'].map(month_map)

    plt.figure(figsize=(11, 5.5), dpi=300)
    palette_years = {2018: '#0284c7', 2019: '#0d9488', 2020: '#f59e0b'}
    ax = sns.lineplot(
        data=monthly_rates,
        x='month_abbr',
        y='is_canceled',
        hue='arrival_date_year',
        palette=palette_years,
        marker='o',
        linewidth=2.5,
        markersize=7
    )
    plt.title('Monthly Cancellation Rates Across Operational Years (2018 - 2020)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Arrival Month', fontsize=12, labelpad=10)
    plt.ylabel('Cancellation Rate', fontsize=12, labelpad=10)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.legend(title='Operating Year', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'monthly_cancellation_trend.png'))
    plt.close()

    # 2. Average Lead Time by Reservation Status
    lead_time_df = hotel_data.groupby('reservation_status')['lead_time'].mean().reset_index().sort_values('lead_time')
    plt.figure(figsize=(8.5, 5), dpi=300)
    ax = sns.barplot(
        data=lead_time_df,
        x='reservation_status',
        y='lead_time',
        palette=['#10b981', '#f59e0b', '#ef4444']
    )
    plt.title('Average Booking Lead Time by Final Reservation Status', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Reservation Status', fontsize=11)
    plt.ylabel('Average Lead Time (Days)', fontsize=11)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f} days',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 6),
                    textcoords='offset points')
    plt.ylim(0, lead_time_df['lead_time'].max() * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'lead_time_by_status.png'))
    plt.close()

    # 3. Market Segment Cancellation Rate
    mkt_cancel = hotel_data.groupby('market_segment')['is_canceled'].agg(['mean', 'count']).reset_index()
    mkt_cancel = mkt_cancel[mkt_cancel['count'] > 50].sort_values(by='mean', ascending=False)
    plt.figure(figsize=(9, 5), dpi=300)
    ax = sns.barplot(
        data=mkt_cancel,
        x='market_segment',
        y='mean',
        palette='YlGnBu_r'
    )
    plt.title('Cancellation Rate by Market Segment (Bookings > 50)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Market Segment', fontsize=11)
    plt.ylabel('Cancellation Rate', fontsize=11)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 5),
                    textcoords='offset points')
    plt.xticks(rotation=25, ha='right')
    plt.ylim(0, mkt_cancel['mean'].max() * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'market_segment_cancellation.png'))
    plt.close()

    # 4. Distribution Channel Cancellation Rate
    dist_cancel = hotel_data.groupby('distribution_channel')['is_canceled'].agg(['mean', 'count']).reset_index()
    dist_cancel = dist_cancel[dist_cancel['count'] > 50].sort_values(by='mean', ascending=False)
    plt.figure(figsize=(8.5, 5), dpi=300)
    ax = sns.barplot(
        data=dist_cancel,
        x='distribution_channel',
        y='mean',
        palette='YlGnBu_r'
    )
    plt.title('Cancellation Rate by Distribution Channel', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Distribution Channel', fontsize=11)
    plt.ylabel('Cancellation Rate', fontsize=11)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 5),
                    textcoords='offset points')
    plt.ylim(0, dist_cancel['mean'].max() * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'distribution_channel_cancellation.png'))
    plt.close()

    # 5. Parking Spaces vs Reservation Status Heatmap
    parking_pivot = hotel_data.pivot_table(
        index='required_car_parking_spaces',
        columns='reservation_status',
        values='adr',
        aggfunc='count',
        fill_value=0
    )
    plt.figure(figsize=(7.5, 4.5), dpi=300)
    sns.heatmap(parking_pivot, annot=True, fmt='d', cmap='YlGnBu', cbar=True, linewidths=1)
    plt.title('Booking Volume by Required Parking Spaces & Final Status', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Reservation Status', fontsize=11)
    plt.ylabel('Required Car Parking Spaces', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'parking_spaces_vs_status.png'))
    plt.close()

    # 6. Special Requests vs Cancellation Rate
    spec_req = hotel_data.groupby('total_of_special_requests')['is_canceled'].agg(['mean', 'count']).reset_index()
    spec_req = spec_req[spec_req['count'] >= 20]
    plt.figure(figsize=(8, 4.5), dpi=300)
    ax = sns.barplot(
        data=spec_req,
        x='total_of_special_requests',
        y='mean',
        palette='YlGnBu_r'
    )
    plt.title('Cancellation Probability Decreases as Special Requests Increase', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Total Special Requests', fontsize=11)
    plt.ylabel('Cancellation Probability', fontsize=11)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 5),
                    textcoords='offset points')
    plt.ylim(0, spec_req['mean'].max() * 1.22)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'special_requests_impact.png'))
    plt.close()

    print("   All EDA visualizations saved to assets/.")


def train_and_export_model(hotel_data):
    print("-> Training Random Forest Classifier & Serializing Artifacts...")
    selected_columns = [
        'lead_time',
        'arrival_date_month',
        'country',
        'market_segment',
        'distribution_channel',
        'reserved_room_type',
        'deposit_type',
        'customer_type',
        'is_canceled'
    ]

    subset_data = hotel_data[selected_columns].copy()

    categorical_columns = [
        'arrival_date_month',
        'country',
        'market_segment',
        'distribution_channel',
        'reserved_room_type',
        'deposit_type',
        'customer_type'
    ]

    # Save distinct label encoders for each column
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        subset_data[col] = le.fit_transform(subset_data[col].astype(str))
        label_encoders[col] = le

    X = subset_data.drop(columns='is_canceled')
    y = subset_data['is_canceled']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"   ROC AUC Score: {roc_auc:.4f}")

    # Feature Importance
    importances = rf_model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]
    feature_rankings = [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in sorted_idx
    ]

    print("\n   Top Features:")
    for item in feature_rankings:
        print(f"   - {item['feature']}: {item['importance']:.4f}")

    # Plot 7: Feature Importance
    plt.figure(figsize=(9, 5), dpi=300)
    feat_df = pd.DataFrame(feature_rankings).sort_values('importance', ascending=True)
    ax = plt.barh(feat_df['feature'], feat_df['importance'], color='#0284c7', edgecolor='#0369a1')
    plt.title('Random Forest Feature Importance Hierarchy', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Gini Importance (Weight)', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    for p in ax:
        plt.text(p.get_width() + 0.008, p.get_y() + p.get_height()/2,
                 f"{p.get_width():.4f} ({p.get_width()*100:.1f}%)",
                 va='center', fontsize=10, fontweight='bold', color='#1e293b')
    plt.xlim(0, feat_df['importance'].max() * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'feature_importance.png'))
    plt.close()

    # Plot 8: Confusion Matrix & ROC Curve (Combined Benchmark)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Confirmed (0)', 'Canceled (1)'],
                yticklabels=['Confirmed (0)', 'Canceled (1)'])
    ax1.set_title('Test Set Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    ax1.set_ylabel('Actual Ground Truth', fontsize=11)

    ax2.plot(fpr, tpr, color='#0284c7', lw=2.5, label=f'Random Forest (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--', label='Random Chance Baseline')
    ax2.set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'model_performance_benchmarks.png'))
    plt.close()

    # Save artifacts
    print("-> Saving model files to models/...")
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest_model.joblib'), compress=5)
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'label_encoders.joblib'))

    metrics_payload = {
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importances": feature_rankings,
        "features": list(X.columns),
        "categorical_columns": categorical_columns
    }
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics_payload, f, indent=2)

    print("-> Pipeline completed successfully!")


if __name__ == '__main__':
    data = load_and_preprocess_data()
    generate_eda_figures(data)
    train_and_export_model(data)
