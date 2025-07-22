import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="SEO Traffic Predictor Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-metric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .warning-metric {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚀 SEO Traffic Predictor Pro</h1>', unsafe_allow_html=True)
st.markdown("**Advanced ML-powered tool to predict organic traffic based on backlinks and other SEO factors**")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["Linear Regression", "Polynomial Regression", "Random Forest"],
        help="Choose the prediction algorithm"
    )
    
    # Confidence interval
    show_confidence = st.checkbox("Show Confidence Intervals", value=True)
    
    # Polynomial degree (if polynomial selected)
    poly_degree = 2
    if model_type == "Polynomial Regression":
        poly_degree = st.slider("Polynomial Degree", 2, 4, 2)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    # Sample data option
    if st.button("🎯 Use Sample SEO Data", type="secondary"):
        # Generate realistic sample data
        np.random.seed(42)
        backlinks = np.random.exponential(50, 100) + np.random.normal(0, 10, 100)
        backlinks = np.clip(backlinks, 10, 500).astype(int)
        
        # More realistic traffic with diminishing returns
        base_traffic = backlinks * (2.5 + np.random.normal(0, 0.5, 100))
        domain_authority_factor = np.random.normal(1.2, 0.3, 100)
        content_quality_factor = np.random.normal(1.1, 0.2, 100)
        
        traffic = (base_traffic * domain_authority_factor * content_quality_factor).astype(int)
        traffic = np.clip(traffic, 50, 10000)
        
        sample_df = pd.DataFrame({
            'Backlinks': backlinks,
            'Organic Traffic': traffic,
            'Domain Authority': np.random.randint(20, 80, 100),
            'Content Score': np.random.randint(60, 100, 100)
        })
        
        st.session_state['df'] = sample_df
        st.success("✅ Sample data loaded!")

with col2:
    # File upload
    uploaded_file = st.file_uploader(
        "📂 Upload Your SEO Data", 
        type=["csv"],
        help="CSV should contain: Backlinks, Organic Traffic (+ optional: Domain Authority, Content Score)"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.success("✅ Your data uploaded successfully!")

# Main analysis
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Data validation
    required_cols = ['Backlinks', 'Organic Traffic']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Missing columns. Required: {required_cols}")
        st.stop()
    
    # Data preview
    with st.expander("📊 Data Overview", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Avg Backlinks", f"{df['Backlinks'].mean():.0f}")
        with col3:
            st.metric("Avg Traffic", f"{df['Organic Traffic'].mean():.0f}")
        
        st.dataframe(df.head(10), use_container_width=True)
    
    # Prepare features
    X = df[['Backlinks']].copy()
    y = df['Organic Traffic']
    
    # Add additional features if available
    feature_cols = ['Backlinks']
    if 'Domain Authority' in df.columns:
        X['Domain Authority'] = df['Domain Authority']
        feature_cols.append('Domain Authority')
    if 'Content Score' in df.columns:
        X['Content Score'] = df['Content Score']
        feature_cols.append('Content Score')
    
    # Model training
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
    
    elif model_type == "Polynomial Regression":
        poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
    
    else:  # Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # Display metrics
    st.subheader("📈 Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color_class = "success-metric" if r2 > 0.7 else "warning-metric" if r2 > 0.4 else "metric-card"
        st.markdown(f'<div class="metric-card {color_class}"><h3>{r2:.3f}</h3><p>R² Score</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{mae:.0f}</h3><p>Mean Abs Error</p></div>', unsafe_allow_html=True)
    
    with col3:
        accuracy = max(0, (1 - mae/y.mean()) * 100)
        st.markdown(f'<div class="metric-card success-metric"><h3>{accuracy:.1f}%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
    
    # Visualization
    st.subheader("📊 Interactive Analysis")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Backlinks vs Traffic', 'Prediction Accuracy', 'Residual Analysis', 'Feature Importance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(x=df['Backlinks'], y=df['Organic Traffic'], 
                  mode='markers', name='Actual Data',
                  marker=dict(color='rgba(50, 171, 96, 0.6)', size=8)),
        row=1, col=1
    )
    
    # Prediction line
    sorted_idx = np.argsort(df['Backlinks'])
    fig.add_trace(
        go.Scatter(x=df['Backlinks'].iloc[sorted_idx], y=y_pred[sorted_idx],
                  mode='lines', name='Prediction Line',
                  line=dict(color='red', width=3)),
        row=1, col=1
    )
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=y, y=y_pred, mode='markers',
                  name='Predictions', showlegend=False,
                  marker=dict(color='rgba(255, 127, 14, 0.6)', size=8)),
        row=1, col=2
    )
    
    # Perfect prediction line
    min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', name='Perfect Prediction',
                  line=dict(dash='dash', color='gray'), showlegend=False),
        row=1, col=2
    )
    
    # Residuals
    residuals = y - y_pred
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers',
                  name='Residuals', showlegend=False,
                  marker=dict(color='rgba(219, 64, 82, 0.6)', size=8)),
        row=2, col=1
    )
    
    # Feature importance (for Random Forest)
    if model_type == "Random Forest":
        importance = model.feature_importances_
        fig.add_trace(
            go.Bar(x=feature_cols, y=importance,
                  name='Importance', showlegend=False,
                  marker_color='rgba(55, 128, 191, 0.7)'),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text="Comprehensive SEO Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction tool
    st.subheader("🔮 Traffic Prediction Tool")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input controls
        input_backlinks = st.slider("Target Backlinks", 
                                   int(df['Backlinks'].min()), 
                                   int(df['Backlinks'].max() * 2), 
                                   int(df['Backlinks'].median()))
        
        # Additional inputs if available
        input_da = 50
        input_content = 80
        
        if 'Domain Authority' in df.columns:
            input_da = st.slider("Domain Authority", 1, 100, 50)
        if 'Content Score' in df.columns:
            input_content = st.slider("Content Quality Score", 1, 100, 80)
    
    with col2:
        # Make prediction
        input_data = {'Backlinks': [input_backlinks]}
        
        if 'Domain Authority' in feature_cols:
            input_data['Domain Authority'] = [input_da]
        if 'Content Score' in feature_cols:
            input_data['Content Score'] = [input_content]
            
        input_features = pd.DataFrame(input_data)
        
        # Ensure correct feature order
        input_features = input_features[feature_cols]
        
        if model_type == "Polynomial Regression":
            input_features_poly = poly_features.transform(input_features)
            prediction = model.predict(input_features_poly)[0]
        else:
            prediction = model.predict(input_features)[0]
        
        # Display prediction
        st.markdown(f"""
        <div class="metric-card success-metric" style="margin-top: 2rem;">
            <h2>{prediction:.0f}</h2>
            <p>Predicted Monthly Traffic</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Growth calculation
        current_traffic = df['Organic Traffic'].iloc[-1] if len(df) > 0 else 1000
        growth = ((prediction - current_traffic) / current_traffic) * 100
        growth_icon = "📈" if growth > 0 else "📉"
        st.metric("Growth Potential", f"{growth:+.1f}%", delta=f"{growth_icon}")
    
    # Insights
    st.subheader("💡 SEO Insights & Recommendations")
    
    insights = []
    
    # Model quality insights
    if r2 > 0.8:
        insights.append("🎯 **Excellent Model**: Strong correlation between backlinks and traffic")
    elif r2 > 0.6:
        insights.append("✅ **Good Model**: Moderate correlation - consider additional factors")
    else:
        insights.append("⚠️ **Weak Correlation**: Traffic may depend more on content quality, keywords, or other factors")
    
    # Feature insights
    if model_type == "Random Forest" and len(feature_cols) > 1:
        importance = model.feature_importances_
        most_important = feature_cols[np.argmax(importance)]
        insights.append(f"🔍 **Key Factor**: {most_important} has the highest impact on traffic")
    
    # Traffic potential
    if prediction > current_traffic:
        insights.append(f"🚀 **Growth Opportunity**: Potential to increase traffic by {growth:.1f}% with more backlinks")
    
    # Display insights
    for insight in insights:
        st.markdown(insight)
    
    # Quick tips
    with st.expander("📚 SEO Best Practices", expanded=False):
        st.markdown("""
        - **Quality over Quantity**: Focus on high-authority backlinks from relevant sites
        - **Content First**: Great content naturally attracts quality backlinks
        - **Monitor Progress**: Track both backlinks and resulting traffic changes
        - **Diversify Sources**: Don't rely solely on backlinks - optimize for keywords, user experience
        - **Patience Required**: SEO results typically take 3-6 months to materialize
        """)

else:
    # Welcome state
    st.markdown("""
    ### 🚀 Get Started
    
    **Option 1**: Upload your CSV with columns: `Backlinks`, `Organic Traffic` (and optionally `Domain Authority`, `Content Score`)
    
    **Option 2**: Use our sample data to explore the tool's capabilities
    
    ### 🎯 What You'll Get:
    - Advanced ML predictions (Linear, Polynomial, Random Forest)
    - Interactive visualizations and insights
    - Growth opportunity analysis
    - SEO best practices and recommendations
    """)

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ for SEO professionals. Remember: Correlation ≠ Causation. Use predictions as guidance, not guarantees.*")