# Disaster Assistance Predictive Analysis Web Application

A comprehensive web application for analyzing disaster data and predicting assistance costs using machine learning.

## Features

### 📊 Data Analysis

- **Overview Dashboard**: Key metrics, disaster timeline, and top disaster types
- **Data Exploration**: Deep dive into affected populations, assistance costs, and correlations
- **Actionable Insights**: Cost efficiency analysis, resource planning, and geospatial hotspot identification

### 🤖 Predictive Models

- **Random Forest Regressor**: Best performing model (R²: ~72%, RMSE: ~₱3.1M)
- **Hist Gradient Boosting Regressor**: Alternative ensemble method (R²: ~62%)
- **Linear Regression**: Baseline model (R²: ~54%)

### 💡 Key Insights

- **Cost-Efficiency**: Identify high-cost items for bulk purchasing negotiations
- **Resource Planning**: Disaster-specific item distribution analysis
- **Geospatial Analysis**: Province and municipality hotspot identification
- **Seasonal Patterns**: Quarterly disaster frequency analysis

### 🎯 Predictions

- Interactive prediction interface
- Input disaster characteristics to forecast assistance costs
- Real-time cost estimates with confidence intervals

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your Excel file `disaster_data_latest (1).xlsx` is in the same directory as the app.

2. Run the Streamlit application:

```bash
streamlit run disaster_analysis_app.py
```

3. The app will open in your default web browser at `http://localhost:8501`

## Data Structure

The application expects an Excel file with the following sheets:

- **affected**: Contains disaster impact data (persons/families affected)
- **assistance**: Contains assistance item distribution and costs
- **evacuation**: Contains evacuation center data

## Model Performance

Based on the disaster assistance dataset:

| Algorithm               | R² Score | RMSE          |
| ----------------------- | -------- | ------------- |
| Random Forest Regressor | 0.72     | ₱3,136,046.34 |
| Hist Gradient Boosting  | 0.62     | ₱3,617,372.03 |
| Linear Regression       | 0.54     | ₱3,973,920.82 |

## Features Implemented

✅ Data loading and cleaning (Excel date conversion)
✅ Feature engineering (temporal features, disaster encoding)
✅ Model training and comparison
✅ Interactive visualizations (Plotly)
✅ Cost efficiency analysis
✅ Resource pre-positioning insights
✅ Geospatial hotspot analysis
✅ Real-time predictions

## Navigation

The app includes 5 main sections:

1. **Overview**: Dataset summary and key metrics
2. **Data Exploration**: Detailed data analysis and visualizations
3. **Predictive Models**: Model training, comparison, and performance metrics
4. **Actionable Insights**: Cost efficiency, resource planning, and geospatial analysis
5. **Predictions**: Interactive prediction interface

## Technical Details

- **Framework**: Streamlit (Python web framework)
- **ML Libraries**: scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

## Notes

- The app automatically handles Excel serial date conversion
- Missing values are handled appropriately for each analysis
- Models are cached for faster performance
- All visualizations are interactive and exportable
