import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Restaurant Recommender", page_icon="🍽️", layout="wide")

@st.cache_data
def load_model():
    with open('restaurant_model.pkl', 'rb') as f:
        return pickle.load(f)

def get_recommendations(user_id, user_item_matrix, top_k=5):
    if user_id not in user_item_matrix.index:
        return []
    
    # Calculate user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    # Get similar users
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:11]
    similar_users = similar_users[similar_users > 0.1]
    
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
    
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_k]

def main():
    st.title("🍽️ Restaurant Recommendation System")
    
    try:
        model_data = load_model()
        df = model_data['df']
        user_item_matrix = model_data['user_item_matrix']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Get Recommendations")
            
            # User selection
            users = df['reviewerId'].unique()
            selected_user = st.selectbox("Select User ID:", users)
            
            # Number of recommendations
            num_recs = st.slider("Number of recommendations:", 1, 10, 5)
            
            if st.button("Get Recommendations", type="primary"):
                recommendations = get_recommendations(selected_user, user_item_matrix, num_recs)
                
                if recommendations:
                    st.success(f"Top {len(recommendations)} recommendations:")
                    
                    for i, (restaurant, score) in enumerate(recommendations, 1):
                        with st.container():
                            st.write(f"**{i}. {restaurant}**")
                            st.write(f"Predicted Rating: {score:.2f}")
                            
                            # Restaurant stats
                            restaurant_data = df[df['title'] == restaurant]
                            if not restaurant_data.empty:
                                avg_rating = restaurant_data['stars'].mean()
                                num_reviews = len(restaurant_data)
                                st.write(f"Average Rating: {avg_rating:.2f} | Reviews: {num_reviews}")
                            st.divider()
                else:
                    st.warning("No recommendations found for this user.")
        
        with col2:
            st.subheader("Dataset Info")
            st.metric("Total Users", df['reviewerId'].nunique())
            st.metric("Total Restaurants", df['title'].nunique())
            st.metric("Total Reviews", len(df))
            st.metric("Avg Rating", f"{df['stars'].mean():.2f}")
            
            # User's review history
            if 'selected_user' in locals():
                st.subheader("User's Previous Reviews")
                user_reviews = df[df['reviewerId'] == selected_user]
                if not user_reviews.empty:
                    for _, review in user_reviews.head(5).iterrows():
                        st.write(f"**{review['title']}**: {review['stars']}⭐")
        
        # Analytics section
        st.subheader("Dataset Analytics")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Top rated restaurants
            top_restaurants = df.groupby('title')['stars'].agg(['mean', 'count'])
            top_restaurants = top_restaurants[top_restaurants['count'] >= 3]
            top_restaurants = top_restaurants.sort_values('mean', ascending=False).head(10)
            
            st.write("**Top Rated Restaurants:**")
            for restaurant, data in top_restaurants.iterrows():
                st.write(f"{restaurant}: {data['mean']:.2f}⭐ ({data['count']} reviews)")
        
        with col4:
            # Most reviewed restaurants
            most_reviewed = df['title'].value_counts().head(10)
            
            st.write("**Most Reviewed Restaurants:**")
            for restaurant, count in most_reviewed.items():
                avg_rating = df[df['title'] == restaurant]['stars'].mean()
                st.write(f"{restaurant}: {count} reviews ({avg_rating:.2f}⭐)")
    
    except FileNotFoundError:
        st.error("Model file 'restaurant_model.pkl' not found!")
        st.info("Please ensure the model file is in the same directory as this app.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()
