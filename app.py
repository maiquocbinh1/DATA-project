"""
Streamlit Web App - Hệ thống gợi ý phim TMDB
HYBRID RECOMMENDATION SYSTEM - Giống Netflix
Author: Final Project - Recommendation System
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests  # Thêm để lấy ảnh poster từ TMDB API
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

# ===== HÀM LẤY ẢNH POSTER TỪ TMDB API =====
@st.cache_data
def fetch_poster(movie_id):
    """
    Lấy ảnh poster phim từ TMDB API
    Args:
        movie_id: ID phim trên TMDB
    Returns:
        str: URL của ảnh poster
    """
    try:
        # API Key miễn phí của TMDB
        api_key = "c7ec19ffdd3279641fb606d19ceb9bb1"
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=vi-VN"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return full_path
        else:
            # Ảnh mặc định nếu không tìm thấy
            return "https://via.placeholder.com/500x750?text=No+Poster"
    except:
        # Ảnh mặc định nếu lỗi (mất mạng, timeout...)
        return "https://via.placeholder.com/500x750?text=No+Poster"

# ===== HÀM GỢI Ý PHIM =====

# 1. Content-Based (1 phim)
def get_recommendations(title, cosine_sim, indices, movies_data, top_n=10):
    """Content-Based Filtering: Gợi ý dựa trên 1 phim"""
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

# 2. Personalized (nhiều phim - User Profile)
def get_personalized_recommendations(selected_titles, cosine_sim, indices, movies_data, top_n=10):
    """Personalized: Tạo User Profile từ nhiều phim yêu thích"""
    try:
        movie_indices = []
        for title in selected_titles:
            if title in indices.index:
                movie_indices.append(indices[title])
        
        if len(movie_indices) == 0:
            return None
        
        # Tạo User Profile bằng cách cộng dồn similarity
        total_scores = np.zeros(cosine_sim.shape[0])
        for idx in movie_indices:
            total_scores += cosine_sim[idx]
        total_scores = total_scores / len(movie_indices)
        
        # Sắp xếp và lọc
        sim_scores = list(enumerate(total_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [x for x in sim_scores if x[0] not in movie_indices]
        
        top_scores = sim_scores[:top_n]
        top_indices = [i[0] for i in top_scores]
        
        result = movies_data.iloc[top_indices].copy()
        result['personalization_score'] = [score[1] for score in top_scores]
        
        return result
        
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        return None

# 3. HYBRID (Content + Personalized + Popularity)
def get_hybrid_recommendations(selected_titles, cosine_sim, indices, movies_data, top_n=10,
                                content_weight=0.4, personalized_weight=0.4, popularity_weight=0.2):
    """HYBRID System: Kết hợp Content + Personalized + Popularity như Netflix"""
    try:
        movie_indices = []
        for title in selected_titles:
            if title in indices.index:
                movie_indices.append(indices[title])
        
        if len(movie_indices) == 0:
            return None
        
        # Chuẩn hóa trọng số
        total_weight = content_weight + personalized_weight + popularity_weight
        if not np.isclose(total_weight, 1.0):
            content_weight /= total_weight
            personalized_weight /= total_weight
            popularity_weight /= total_weight
        
        # Personalized scores
        personalized_scores = np.zeros(cosine_sim.shape[0])
        for idx in movie_indices:
            personalized_scores += cosine_sim[idx]
        personalized_scores = personalized_scores / len(movie_indices)
        
        # Content scores (tương tự)
        content_scores = personalized_scores.copy()
        
        # Popularity scores (từ dữ liệu đã chuẩn hóa)
        popularity_scores = movies_data['vote_avg_scaled'].values * 0.7 + \
                           movies_data['popularity_scaled'].values * 0.3
        
        # Chuẩn hóa về [0, 1]
        content_scores_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
        personalized_scores_norm = (personalized_scores - personalized_scores.min()) / (personalized_scores.max() - personalized_scores.min() + 1e-8)
        
        # HYBRID SCORE
        hybrid_scores = (
            content_weight * content_scores_norm +
            personalized_weight * personalized_scores_norm +
            popularity_weight * popularity_scores
        )
        
        # Sắp xếp
        scored_items = list(enumerate(hybrid_scores))
        scored_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
        scored_items = [x for x in scored_items if x[0] not in movie_indices]
        
        top_items = scored_items[:top_n]
        top_indices = [i[0] for i in top_items]
        
        result = movies_data.iloc[top_indices].copy()
        result['hybrid_score'] = [item[1] for item in top_items]
        result['content_component'] = [content_scores_norm[item[0]] for item in top_items]
        result['personalized_component'] = [personalized_scores_norm[item[0]] for item in top_items]
        result['popularity_component'] = [popularity_scores[item[0]] for item in top_items]
        
        return result
        
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        return None

# Load model và dữ liệu
data = load_model()
movies_data = data['movies_data']
cosine_sim = data['cosine_sim']
indices = data['indices']

# Khởi tạo session state cho lịch sử tìm kiếm
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Khởi tạo state cho việc hiển thị lịch sử
if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# Header
st.markdown('<h1 class="main-header">Movie Recommender System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Tìm kiếm phim yêu thích và nhận gợi ý phim tương tự dựa trên nội dung</p>', unsafe_allow_html=True)

# Hiển thị lịch sử tìm kiếm ở main area (nếu đã bật từ sidebar)
if st.session_state.show_history:
    st.divider()
    
    # Header với nút quay lại
    col_title, col_back = st.columns([8, 2])
    with col_title:
        if len(st.session_state.search_history) == 0:
            st.subheader("📜 Lịch sử tìm kiếm")
        else:
            st.subheader(f"📜 Lịch sử tìm kiếm ({len(st.session_state.search_history)} lần)")
    with col_back:
        if st.button("← Quay lại", width="stretch"):
            st.session_state.show_history = False
            st.rerun()
    
    st.divider()
    
    if len(st.session_state.search_history) == 0:
        st.info("Chưa có lịch sử tìm kiếm. Hãy tìm kiếm phim để tạo lịch sử!")
    else:
        
        st.divider()
        
        # Hiển thị lịch sử (đảo ngược để mới nhất ở trên)
        for i, history in enumerate(reversed(st.session_state.search_history[-10:])):  # Chỉ hiển thị 10 lần gần nhất
            with st.container():
                col_num, col_info = st.columns([1, 9])
                with col_num:
                    st.markdown(f"### #{len(st.session_state.search_history) - i}")
                with col_info:
                    st.markdown(f"""
                    **🕐 Thời gian:** {history['time']}  
                    **🎯 Mode:** {history['mode']}  
                    **🎬 Phim đã chọn:** {', '.join(history['movies'][:3])}{'...' if len(history['movies']) > 3 else ''}  
                    **📊 Số kết quả:** {history['num_results']} phim
                    """)
                st.divider()
        
        # Nút xóa lịch sử
        col_delete, col_spacer = st.columns([2, 8])
        with col_delete:
            if st.button("🗑️ Xóa toàn bộ lịch sử", width="stretch"):
                st.session_state.search_history = []
                st.session_state.show_history = False
                st.rerun()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/200/000000/clapperboard.png", width=150)
    st.metric("Tổng số phim", f"{len(movies_data):,}")
    
    # Button lịch sử tìm kiếm (toggle)
    history_count = len(st.session_state.search_history)
    button_label = f"📜 Lịch sử ({history_count})" if history_count > 0 else "📜 Lịch sử"
    
    if st.button(button_label, width="stretch"):
        st.session_state.show_history = not st.session_state.show_history
        st.rerun()  # Reload ngay lập tức
    
    st.divider()

    st.subheader("Chọn chế độ gợi ý")
    recommendation_mode = st.radio(
        "Mode:",
        ["Content-Based", "Personalized", "HYBRID (Netflix-style)"],
        index=2,  # Mặc định là Hybrid
        help="""
        • Content-Based: Dựa trên 1 phim
        • Personalized: Dựa trên nhiều phim (User Profile)
        • HYBRID: Kết hợp tất cả (Khuyên dùng!)
        """
    )

    # [THÊM MỚI] Context-aware: Chọn tâm trạng (đặt sau mode để đúng luồng báo cáo)
    st.divider()
    st.subheader("🎭 Lọc theo Tâm trạng (Context-aware)")
    selected_mood = st.selectbox(
        "Hôm nay bạn thế nào?",
        [
            "Tất cả (Mặc định)",
            "😄 Vui vẻ / Hài hước",
            "😢 Buồn / Sâu lắng",
            "😱 Hồi hộp / Gay cấn",
            "😎 Hành động / Kịch tính",
        ],
        help="Hệ thống sẽ lọc kết quả dựa trên cảm xúc hiện tại của bạn",
        index=0,
    )
    
    # Nếu chọn HYBRID, cho phép điều chỉnh trọng số
    if "HYBRID" in recommendation_mode:
        st.divider()
        st.subheader("Điều chỉnh trọng số")
        st.caption("Tổng = 100%")
        
        content_w = st.slider("Content", 0, 100, 40, 5, help="Độ tương đồng nội dung") / 100
        personalized_w = st.slider("Personalized", 0, 100, 40, 5, help="Phù hợp với sở thích") / 100
        popularity_w = st.slider("Popularity", 0, 100, 20, 5, help="Đánh giá của cộng đồng") / 100
        
        total = content_w + personalized_w + popularity_w
        if not np.isclose(total, 1.0):
            st.warning(f"⚠️ Tổng = {total*100:.0f}% (sẽ tự động chuẩn hóa về 100%)")
    else:
        content_w, personalized_w, popularity_w = 0.4, 0.4, 0.2

# Chỉ hiển thị main content khi KHÔNG xem lịch sử
if not st.session_state.show_history:
    # Main content
    col1, col2 = st.columns([3, 1])

    with col1:
        # Tìm kiếm phim - thay đổi theo mode
        if "Content-Based" in recommendation_mode:
            # Single select cho Content-Based
            search_option = st.selectbox(
                "Chọn phim yêu thích của bạn:",
                options=[""] + sorted(movies_data['title'].tolist()),
                index=0
            )
            selected_movies = [search_option] if search_option != "" else []
        else:
            # Multi-select cho Personalized và Hybrid
            selected_movies = st.multiselect(
                "Chọn 3-5 phim bạn yêu thích:",
                options=sorted(movies_data['title'].tolist()),
                default=[],
                help="Chọn nhiều phim để hệ thống hiểu rõ GU của bạn hơn"
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
    button_label = "Tìm phim tương tự" if "Content-Based" in recommendation_mode else "Tìm phim phù hợp với tôi"
    if st.button(button_label, width="stretch"):
        if len(selected_movies) == 0:
            st.warning("⚠️ Vui lòng chọn ít nhất một phim!")
        elif "Content-Based" not in recommendation_mode and len(selected_movies) < 2:
            st.warning("⚠️ Personalized/Hybrid mode cần ít nhất 2 phim để hiểu GU của bạn!")
        else:
            spinner_text = f"Đang phân tích {'GU' if len(selected_movies) > 1 else 'phim'} của bạn..."
            with st.spinner(spinner_text):
                # Hiển thị thông tin các phim được chọn
                if len(selected_movies) == 1:
                    selected_movie = movies_data[movies_data['title'] == selected_movies[0]].iloc[0]
                    st.success(f"✅ Đã chọn: **{selected_movies[0]}**")
                    
                    # Hiển thị poster và thông tin
                    col_poster_main, col_info_main = st.columns([1, 3])
                
                    with col_poster_main:
                        poster_url = fetch_poster(selected_movie['id'])
                        st.image(poster_url, width="stretch")
                    
                    with col_info_main:
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
                else:
                    st.success(f"✅ Đã chọn {len(selected_movies)} phim: **{', '.join(selected_movies[:3])}**{'...' if len(selected_movies) > 3 else ''}")
                    st.info(f"🤖 Hệ thống đang phân tích GU TỔNG HỢP của bạn từ {len(selected_movies)} phim này...")
                
                st.divider()
                
                # Gọi hàm tương ứng với mode
                if "Content-Based" in recommendation_mode:
                    recommendations = get_recommendations(
                        selected_movies[0], 
                        cosine_sim, 
                        indices, 
                        movies_data, 
                        top_n=num_recommendations
                    )
                    score_column = 'similarity_score'
                    score_label = "🎯 Match"
                elif "Personalized" in recommendation_mode:
                    recommendations = get_personalized_recommendations(
                        selected_movies,
                        cosine_sim,
                        indices,
                        movies_data,
                        top_n=num_recommendations
                    )
                    score_column = 'personalization_score'
                    score_label = "👤 Personal Match"
                else:  # HYBRID
                    recommendations = get_hybrid_recommendations(
                        selected_movies,
                        cosine_sim,
                        indices,
                        movies_data,
                        top_n=num_recommendations,
                        content_weight=content_w,
                        personalized_weight=personalized_w,
                        popularity_weight=popularity_w
                    )
                    score_column = 'hybrid_score'
                    score_label = "⭐ Hybrid Score"

                # [THÊM MỚI] Context-Aware Logic: Lọc kết quả theo tâm trạng (Post-filtering)
                # (sau khi có recommendations và trước if recommendations is None:)
                if recommendations is not None and "Tất cả" not in selected_mood:
                    recommendations_before_filter = recommendations

                    # Logic: Map tâm trạng sang các từ khóa thể loại (Genres)
                    if "Vui vẻ" in selected_mood:
                        # Giữ lại phim có chữ Comedy, Family hoặc Animation
                        recommendations = recommendations[
                            recommendations["genres_clean"].str.contains(
                                r"Comedy|Family|Animation", case=False, na=False
                            )
                        ]
                    elif "Buồn" in selected_mood:
                        recommendations = recommendations[
                            recommendations["genres_clean"].str.contains(
                                r"Drama|Romance", case=False, na=False
                            )
                        ]
                    elif "Hồi hộp" in selected_mood:
                        recommendations = recommendations[
                            recommendations["genres_clean"].str.contains(
                                r"Horror|Thriller|Mystery", case=False, na=False
                            )
                        ]
                    elif "Hành động" in selected_mood:
                        recommendations = recommendations[
                            recommendations["genres_clean"].str.contains(
                                r"Action|Adventure|Crime", case=False, na=False
                            )
                        ]

                    # Nếu lọc xong mà hết phim thì hiển thị cảnh báo + fallback về danh sách gốc
                    if recommendations.empty:
                        st.warning(
                            f"⚠️ Không tìm thấy phim phù hợp tâm trạng '{selected_mood}' trong top gợi ý này. Đang hiển thị tất cả..."
                        )
                        recommendations = recommendations_before_filter

                if recommendations is None:
                    st.error("❌ Không thể tìm thấy phim trong cơ sở dữ liệu.")
                else:
                    # Lưu vào lịch sử tìm kiếm
                    from datetime import datetime
                    history_entry = {
                        'time': datetime.now().strftime("%d/%m/%Y %H:%M"),
                        'mode': recommendation_mode,
                        'movies': selected_movies,
                        'num_results': num_recommendations
                    }
                    st.session_state.search_history.append(history_entry)
                    
                    # Header tùy theo mode
                    if "HYBRID" in recommendation_mode:
                        st.subheader(f"Top {num_recommendations} phim dành riêng cho bạn (HYBRID):")
                    elif "Personalized" in recommendation_mode:
                        st.subheader(f"Top {num_recommendations} phim phù hợp với GU của bạn:")
                    else:
                        st.subheader(f"Top {num_recommendations} phim tương tự:")
                    
                    # Hiển thị từng phim gợi ý
                    for rank, (idx, row) in enumerate(recommendations.iterrows(), start=1):
                        with st.container():
                            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                            
                            col_rank, col_poster, col_content = st.columns([0.7, 1.5, 7.8])
                            
                            with col_rank:
                                st.markdown(f"<div style='color: #FF6B6B; font-size: 1.8rem; font-weight: bold; white-space: nowrap;'>#{rank}</div>", unsafe_allow_html=True)
                            
                            with col_poster:
                                # Hiển thị ảnh poster
                                poster_url = fetch_poster(row['id'])
                                st.image(poster_url, width="stretch")
                            
                            with col_content:
                                st.markdown(f"### {row['title']}")
                                
                                # Hiển thị metrics tùy theo mode
                                if "HYBRID" in recommendation_mode:
                                    col_a, col_b, col_c, col_d = st.columns(4)
                                    col_a.write(f"⭐ **{row['vote_average']:.1f}**/10")
                                    col_b.write(f"👥 **{int(row['vote_count']):,}** votes")
                                    col_c.write(f"🎯 **Match:** {row[score_column]:.3f}")
                                    col_d.write(f"📈 **Pop:** {row['popularity']:.1f}")
                                    
                                    # Thêm chi tiết các components
                                    with st.expander("🔍 Xem chi tiết điểm số"):
                                        comp_cols = st.columns(3)
                                        comp_cols[0].metric("Content", f"{row['content_component']:.2%}")
                                        comp_cols[1].metric("Personalized", f"{row['personalized_component']:.2%}")
                                        comp_cols[2].metric("Popularity", f"{row['popularity_component']:.2%}")
                                else:
                                    col_a, col_b, col_c, col_d = st.columns(4)
                                    col_a.write(f"⭐ **{row['vote_average']:.1f}**/10")
                                    col_b.write(f"👥 **{int(row['vote_count']):,}** votes")
                                    col_c.write(f"🎯 **Match:** {row[score_column]:.2%}")
                                    col_d.write(f"📈 **Pop:** {row['popularity']:.1f}")
                                
                                st.write(f"**Thể loại:** {row['genres_clean']}")
                                
                                if pd.notna(row['overview']) and row['overview']:
                                    with st.expander("📖 Đọc tóm tắt"):
                                        st.write(row['overview'])
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Download recommendations
                    st.divider()
                    
                    # Chọn cột phù hợp để export
                    export_cols = ['title', 'vote_average', 'vote_count', score_column, 'genres_clean']
                    if "HYBRID" in recommendation_mode:
                        export_cols.extend(['content_component', 'personalized_component', 'popularity_component'])
                    
                    csv = recommendations[export_cols].to_csv(index=False)
                    filename = f"recommendations_{'_'.join(selected_movies[:2]).replace(' ', '_')}.csv"
                    st.download_button(
                        label="Tải xuống danh sách gợi ý (CSV)",
                        data=csv,
                        file_name=filename,
                        mime="text/csv",
                        width="stretch"
                    )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem;'>
    <p><strong>Final Project - Recommendation System</strong></p>
    <p>Dataset: TMDB 5000 Movies | Model: <strong>HYBRID System (Netflix-style)</strong></p>
    <p>Content-Based + Personalized + Popularity</p>
    <p>Made with ❤️ using Streamlit & scikit-learn</p>
</div>
""", unsafe_allow_html=True)

