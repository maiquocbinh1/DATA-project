"""
Streamlit Web App - Hệ thống gợi ý phim TMDB
Author: Final Project - Recommendation System
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Cấu hình trang
st.set_page_config(
    page_title="🎬 Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 3rem;
    }
    .movie-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #FF6B6B;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #E74C3C;
    }
</style>
""", unsafe_allow_html=True)

# Hàm load model
@st.cache_resource
def load_model():
    try:
        with open('movie_recommender_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("❌ Không tìm thấy file model! Vui lòng chạy notebook trước để tạo model.")
        st.stop()

# Hàm gợi ý phim
def get_recommendations(title, cosine_sim, indices, movies_data, top_n=10):
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        
        result = movies_data.iloc[movie_indices].copy()
        result['similarity_score'] = [score[1] for score in sim_scores]
        return result
    except KeyError:
        return None

# Load model và dữ liệu
data = load_model()
movies_data = data['movies_data']
cosine_sim = data['cosine_sim']
indices = data['indices']

# Header
st.markdown('<h1 class="main-header">🎬 Movie Recommender System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Tìm kiếm phim yêu thích và nhận gợi ý phim tương tự dựa trên nội dung</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/200/000000/clapperboard.png", width=150)
    st.title("📊 Thông tin hệ thống")
    st.metric("Tổng số phim", f"{len(movies_data):,}")
    st.metric("Mô hình", "Content-Based Filtering")
    st.metric("Vector hóa", "TF-IDF")
    
    st.divider()
    
    st.subheader("🎯 Hướng dẫn sử dụng")
    st.markdown("""
    1. Chọn hoặc nhập tên phim bạn yêu thích
    2. Điều chỉnh số lượng gợi ý (1-20)
    3. Nhấn nút **'Tìm phim tương tự'**
    4. Xem danh sách phim được gợi ý
    """)
    
    st.divider()
    
    st.info("💡 **Tip**: Hệ thống gợi ý dựa trên thể loại, nội dung, diễn viên và đạo diễn của phim.")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    # Tìm kiếm phim
    search_option = st.selectbox(
        "🔍 Chọn phim yêu thích của bạn:",
        options=[""] + sorted(movies_data['title'].tolist()),
        index=0
    )

with col2:
    # Số lượng gợi ý
    num_recommendations = st.slider(
        "Số lượng gợi ý:",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )

# Nút tìm kiếm
if st.button("🎥 Tìm phim tương tự", use_container_width=True):
    if search_option == "":
        st.warning("⚠️ Vui lòng chọn một phim trước!")
    else:
        with st.spinner(f"Đang tìm phim tương tự với '{search_option}'..."):
            # Hiển thị thông tin phim được chọn
            selected_movie = movies_data[movies_data['title'] == search_option].iloc[0]
            
            st.success(f"✅ Đã chọn: **{search_option}**")
            
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            
            with col_info1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("⭐ Rating", f"{selected_movie['vote_average']:.1f}/10")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_info2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("👥 Votes", f"{int(selected_movie['vote_count']):,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_info3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📈 Popularity", f"{selected_movie['popularity']:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_info4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                runtime = selected_movie['runtime'] if pd.notna(selected_movie['runtime']) else 0
                st.metric("⏱️ Runtime", f"{int(runtime)} min")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("**🎭 Thể loại:** " + selected_movie['genres_clean'])
            
            if pd.notna(selected_movie['overview']) and selected_movie['overview']:
                with st.expander("📝 Xem tóm tắt nội dung"):
                    st.write(selected_movie['overview'])
            
            st.divider()
            
            # Lấy gợi ý
            recommendations = get_recommendations(
                search_option, 
                cosine_sim, 
                indices, 
                movies_data, 
                top_n=num_recommendations
            )
            
            if recommendations is None:
                st.error("❌ Không thể tìm thấy phim này trong cơ sở dữ liệu.")
            else:
                st.subheader(f"🎬 Top {num_recommendations} phim tương tự:")
                
                # Hiển thị từng phim gợi ý
                for idx, row in recommendations.iterrows():
                    with st.container():
                        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                        
                        col_rank, col_content = st.columns([1, 9])
                        
                        with col_rank:
                            rank = recommendations.index.get_loc(idx) + 1
                            st.markdown(f"<h2 style='color: #FF6B6B;'>#{rank}</h2>", unsafe_allow_html=True)
                        
                        with col_content:
                            st.markdown(f"### {row['title']}")
                            
                            col_a, col_b, col_c, col_d = st.columns(4)
                            col_a.write(f"⭐ **{row['vote_average']:.1f}**/10")
                            col_b.write(f"👥 **{int(row['vote_count']):,}** votes")
                            col_c.write(f"🎯 **{row['similarity_score']:.2%}** match")
                            col_d.write(f"📈 Pop: **{row['popularity']:.1f}**")
                            
                            st.write(f"**Thể loại:** {row['genres_clean']}")
                            
                            if pd.notna(row['overview']) and row['overview']:
                                with st.expander("📖 Đọc tóm tắt"):
                                    st.write(row['overview'])
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Download recommendations
                st.divider()
                csv = recommendations[['title', 'vote_average', 'vote_count', 'similarity_score', 'genres_clean']].to_csv(index=False)
                st.download_button(
                    label="📥 Tải xuống danh sách gợi ý (CSV)",
                    data=csv,
                    file_name=f"recommendations_{search_option.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem;'>
    <p>🎓 <strong>Final Project - Recommendation System</strong></p>
    <p>📊 Dataset: TMDB 5000 Movies | 🤖 Model: Content-Based Filtering</p>
    <p>Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)

