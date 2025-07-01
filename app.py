import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# Page configuration
st.set_page_config(
    page_title="💕 RestaurantDoll",
    page_icon="💕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@400;700&family=Quicksand:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Quicksand', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #FFB6C1 0%, #FFC0CB 50%, #FF69B4 100%);
    padding: 2rem;
    border-radius: 25px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 15px 35px rgba(255,182,193,0.4);
    border: 2px solid rgba(255,255,255,0.3);
}

.main-header h1 {
    color: white;
    font-family: 'Dancing Script', cursive;
    font-size: 3.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(255,105,180,0.3);
}

.main-header p {
    color: white;
    font-size: 1.2rem;
    margin-top: 0.5rem;
    opacity: 0.95;
    font-weight: 400;
}

.metric-card {
    background: linear-gradient(135deg, #FF69B4 0%, #FFB6C1 100%);
    padding: 1.5rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 30px rgba(255,105,180,0.3);
    transition: transform 0.3s ease;
    border: 2px solid rgba(255,255,255,0.2);
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 15px 40px rgba(255,105,180,0.4);
}

.recommendation-card {
    background: linear-gradient(135deg, #FFB6C1 0%, #FFC0CB 100%);
    padding: 2rem;
    border-radius: 25px;
    margin: 1.5rem 0;
    color: white;
    box-shadow: 0 12px 30px rgba(255,182,193,0.3);
    transition: all 0.3s ease;
    border: 2px solid rgba(255,255,255,0.2);
}

.recommendation-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 18px 45px rgba(255,182,193,0.4);
}

.algorithm-badge {
    background: rgba(255,255,255,0.3);
    padding: 0.4rem 1.2rem;
    border-radius: 25px;
    font-size: 0.8rem;
    margin-bottom: 1rem;
    display: inline-block;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.4);
}

.stButton > button {
    background: linear-gradient(135deg, #FF69B4 0%, #FFB6C1 100%);
    color: white;
    border: none;
    border-radius: 30px;
    padding: 1rem 2.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px rgba(255,105,180,0.3);
    font-family: 'Quicksand', sans-serif;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(255,105,180,0.4);
}

.restaurant-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    font-family: 'Dancing Script', cursive;
}

.restaurant-score {
    font-size: 2.2rem;
    font-weight: 700;
    color: #FF1493;
    text-shadow: 1px 1px 2px rgba(255,20,147,0.3);
}

.success-message {
    background: linear-gradient(135deg, #FF69B4 0%, #FFB6C1 100%);
    padding: 1.5rem;
    border-radius: 20px;
    color: white;
    margin: 1.5rem 0;
    border: 2px solid rgba(255,255,255,0.3);
    box-shadow: 0 10px 25px rgba(255,105,180,0.3);
}

.info-card {
    background: linear-gradient(135deg, #FFB6C1 0%, #FFC0CB 100%);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    margin: 1.5rem 0;
    box-shadow: 0 12px 30px rgba(255,182,193,0.3);
    border: 2px solid rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

def cosine_similarity_manual(matrix):
    """Manual cosine similarity calculation"""
    dot_product = np.dot(matrix, matrix.T)
    norms = np.linalg.norm(matrix, axis=1)
    return dot_product / np.outer(norms, norms)

@st.cache_data
def load_model():
    try:
        with open('restaurant_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

def create_animated_metrics(df):
    """Create animated metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Lovely Users", df['reviewerId'].nunique(), "darlings"),
        ("Sweet Spots", df['title'].nunique(), "cafes"),
        ("Love Notes", len(df), "reviews"),
        ("Stars Given", f"{df['stars'].mean():.1f}", "average")
    ]
    
    for i, (col, (title, value, suffix)) in enumerate(zip([col1, col2, col3, col4], metrics)):
        with col:
            st.markdown(f"""
            <div class="metric-card animated-element" style="animation-delay: {i*0.2}s;">
                <h3 style="margin:0; font-size:1rem;">{title}</h3>
                <p style="margin:0; font-size:2rem; font-weight:700;">{value}</p>
            </div>
            """, unsafe_allow_html=True)

def get_recommendations(user_id, user_item_matrix, top_k=5):
    if user_id not in user_item_matrix.index:
        return []
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text('Finding your taste preferences...')
    progress_bar.progress(25)
    time.sleep(0.5)
    
    # Calculate similarity
    user_similarity = cosine_similarity_manual(user_item_matrix.values)
    user_sim_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    status_text.text('Discovering similar food lovers...')
    progress_bar.progress(50)
    time.sleep(0.5)
    
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:11]
    similar_users = similar_users[similar_users > 0.1]
    
    status_text.text('Creating perfect matches...')
    progress_bar.progress(75)
    time.sleep(0.5)
    
    # Generate recommendations
    user_ratings = user_item_matrix.loc[user_id]
    recommendations = {}
    
    for restaurant in user_item_matrix.columns:
        if user_ratings[restaurant] == 0:
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user, similarity in similar_users.items():
                rating = user_item_matrix.loc[similar_user, restaurant]
                if rating > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                recommendations[restaurant] = weighted_sum / similarity_sum
    
    status_text.text('Finalizing your perfect picks...')
    progress_bar.progress(100)
    time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_k]

def display_recommendations(recommendations, df, algorithm="AI Hybrid"):
    """Display recommendations with beautiful cards"""
    if not recommendations:
        st.warning("Oopsie! No sweet recommendations found for this cutie.")
        return
    
    st.markdown(f"""
    <div class="success-message">
        <h3 style="margin:0;">Your Top {len(recommendations)} Perfect Matches ({algorithm})</h3>
        <p style="margin:0.5rem 0 0 0;">Handpicked just for you, sweetie!</p>
    </div>
    """, unsafe_allow_html=True)
    
    for i, (restaurant, score) in enumerate(recommendations):
        # Get restaurant details
        restaurant_data = df[df['title'] == restaurant]
        avg_rating = restaurant_data['stars'].mean() if not restaurant_data.empty else 0
        num_reviews = len(restaurant_data)
        
        # Create recommendation card
        st.markdown(f"""
        <div class="recommendation-card animated-element" style="animation-delay: {i*0.1}s;">
            <div class="algorithm-badge">{algorithm} Magic</div>
            <div class="restaurant-title">#{i+1} {restaurant}</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 1rem; opacity: 0.9;">
                        Rating: {avg_rating:.1f} hearts | {num_reviews} lovely reviews
                    </div>
                </div>
                <div class="restaurant-score">{score:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_interactive_charts(df):
    """Create visualization dashboard using Streamlit native charts"""
    st.markdown("""
    <div class="info-card">
        <h2 style="margin:0;">Sweet Analytics Dashboard</h2>
        <p style="margin:0.5rem 0 0 0;">Discover the cutest insights from our data</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        st.subheader("Rating Distribution")
        rating_counts = df['stars'].value_counts().sort_index()
        st.bar_chart(rating_counts)
    
    with col2:
        # Top restaurants
        st.subheader("Top Rated Sweet Spots")
        top_restaurants = df.groupby('title')['stars'].agg(['mean', 'count'])
        top_restaurants = top_restaurants[top_restaurants['count'] >= 3]
        top_restaurants = top_restaurants.sort_values('mean', ascending=True).tail(10)
        st.bar_chart(top_restaurants['mean'])

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>RestaurantDoll</h1>
        <p>Your Cute AI Food Companion for Mauritania</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    
    if model_data:
        df = model_data['df']
        user_item_matrix = model_data['user_item_matrix']
        
        # Animated metrics
        create_animated_metrics(df)
        
        # Sidebar
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h2>🎛️ Control Panel</h2>
            </div>
            """, unsafe_allow_html=True)
            
            users = df['reviewerId'].unique()
            selected_user = st.selectbox("👤 Select User", users, key="user_select")
            
            num_recs = st.slider("📊 Number of Recommendations", 1, 10, 5)
            
            algorithm = st.selectbox("🧠 Algorithm", 
                                   ["AI Hybrid", "Collaborative Filtering", "Popularity-Based"])
            
            show_analytics = st.checkbox("📈 Show Analytics Dashboard", value=True)
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "🔍 Explore Restaurants", "👤 User Profile"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("✨ Generate Recommendations", type="primary"):
                    with st.spinner("🤖 AI is thinking..."):
                        if algorithm == "Collaborative Filtering":
                            recommendations = get_recommendations(selected_user, user_item_matrix, num_recs)
                            display_recommendations(recommendations, df, "Collaborative Filtering")
                        elif algorithm == "Popularity-Based":
                            popular = df.groupby('title')['stars'].agg(['mean', 'count'])
                            popular = popular[popular['count'] >= 3].sort_values('mean', ascending=False)
                            recommendations = [(rest, rating) for rest, rating in popular.head(num_recs)['mean'].items()]
                            display_recommendations(recommendations, df, "Popularity-Based")
                        else:
                            recommendations = get_recommendations(selected_user, user_item_matrix, num_recs)
                            display_recommendations(recommendations, df, "AI Hybrid")
            
            with col2:
                st.markdown("""
                <div class="info-card">
                    <h3 style="margin:0;">💡 How it works</h3>
                    <p style="margin:0.5rem 0;">Our AI analyzes your preferences and finds similar users to recommend restaurants you'll love!</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            search_term = st.text_input("🔍 Search Restaurants", placeholder="Type restaurant name...")
            
            if search_term:
                filtered = df[df['title'].str.contains(search_term, case=False, na=False)]
                if not filtered.empty:
                    restaurant_summary = filtered.groupby('title').agg({
                        'stars': ['mean', 'count']
                    }).round(2)
                    restaurant_summary.columns = ['Avg Rating', 'Review Count']
                    
                    for restaurant, data in restaurant_summary.iterrows():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div class="restaurant-title">{restaurant}</div>
                            <div>⭐ {data['Avg Rating']:.1f} | 📝 {data['Review Count']} reviews</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        with tab3:
            user_data = df[df['reviewerId'] == selected_user]
            if not user_data.empty:
                st.markdown(f"""
                <div class="info-card">
                    <h3 style="margin:0;">👤 User Profile: {selected_user}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Reviews Written", len(user_data))
                with col2:
                    st.metric("Avg Rating Given", f"{user_data['stars'].mean():.1f}")
                with col3:
                    st.metric("Restaurants Visited", user_data['title'].nunique())
                
                # User's review history
                st.subheader("📝 Recent Reviews")
                for _, review in user_data.head(5).iterrows():
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                        <strong>{review['title']}</strong> - {'⭐' * int(review['stars'])} ({review['stars']})
                    </div>
                    """, unsafe_allow_html=True)
        
        # Analytics Dashboard
        if show_analytics:
            st.markdown("---")
            create_interactive_charts(df)
            
    else:
        st.markdown("""
        <div class="info-card">
            <h2 style="margin:0;">🚀 Welcome to RestaurantAI!</h2>
            <p style="margin:0.5rem 0 0 0;">Upload your restaurant model to get started with AI-powered recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("📁 Please ensure 'restaurant_model.pkl' is in the app directory")

if __name__ == "__main__":
    main()
