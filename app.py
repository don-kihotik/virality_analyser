import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import openai
import googleapiclient.discovery
import os
import time
import threading
# ngrok import has been removed

# --- Helper Functions ---

def get_secret(key):
    # Streamlit secrets have highest priority, then env variable, then None
    return st.secrets.get(key) if key in st.secrets else os.environ.get(key)

def search_youtube_videos(query, n_results):
    """Searches YouTube for a given query and retrieves metadata for the top N videos."""
    try:
        api_key = get_secret('YOUTUBE_API_KEY')  # Use Streamlit secrets or environment variable
        if not api_key:
            st.error("YouTube API key not found. Please set it in Streamlit secrets.")
            return []
        youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

        search_response = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=n_results
        ).execute()

        video_ids = [item['id']['videoId'] for item in search_response.get('items', [])
                     if item.get('id', {}).get('kind') == 'youtube#video' and 'videoId' in item.get('id', {})]

        if not video_ids:
            return []

        videos_response = youtube.videos().list(
            id=','.join(video_ids),
            part='snippet,statistics'
        ).execute()

        video_data = []
        for item in videos_response.get('items', []):
            snippet = item.get('snippet', {})
            statistics = item.get('statistics', {})
            video_data.append({
                'title': snippet.get('title'),
                'description': snippet.get('description'),
                'tags': snippet.get('tags', []),
                'view_count': statistics.get('viewCount'),
                'like_count': statistics.get('likeCount'),
                'comment_count': statistics.get('commentCount'),
                'published_at': snippet.get('publishedAt'),
                'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url')
            })

        return video_data

    except Exception as e:
        st.error(f"Error searching YouTube videos: {e}")
        return []

def calculate_virality_score(df):
    """Calculates a virality score for each video based on engagement metrics."""
    for col in ['view_count', 'like_count', 'comment_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    scaler = MinMaxScaler()
    if not df[['view_count', 'like_count', 'comment_count']].isnull().all().all():
         df[['view_count_scaled', 'like_count_scaled', 'comment_count_scaled']] = scaler.fit_transform(
            df[['view_count', 'like_count', 'comment_count']]
        )
    else:
        df[['view_count_scaled', 'like_count_scaled', 'comment_count_scaled']] = 0

    df['virality_score'] = (
        df['view_count_scaled'] * 0.5 +
        df['like_count_scaled'] * 0.3 +
        df['comment_count_scaled'] * 0.2
    )
    return df

def perform_cluster_analysis(df):
    """Performs clustering on relevant features to group similar videos."""
    features_for_clustering = df[['virality_score', 'view_count_scaled', 'like_count_scaled', 'comment_count_scaled']].copy()
    features_for_clustering.dropna(inplace=True)

    if features_for_clustering.empty:
        df['cluster_label'] = -1
        return df

    n_clusters = min(3, len(features_for_clustering))
    if n_clusters < 1:
         df['cluster_label'] = -1
         return df

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_for_clustering)

    df['cluster_label'] = -1
    df.loc[features_for_clustering.index, 'cluster_label'] = cluster_labels

    return df

def detect_topic_trends(df):
    """Analyzes text data to identify recurring themes or trends."""
    df['text_data'] = df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)).fillna('')

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_data'])
    except ValueError:
        return {"Topic Trends": ["Could not detect topic trends (insufficient text data)."]}

    n_topics = min(5, tfidf_matrix.shape[1] if tfidf_matrix.shape[1] > 0 else 0)
    if n_topics < 1:
         return {"Topic Trends": ["Could not detect topic trends (insufficient features for LDA)."]}

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)

    detected_topics = {}
    feature_names = tfidf_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        detected_topics[f"Topic #{topic_idx}"] = top_keywords

    return detected_topics

def generate_naming_suggestions(df):
    """Generates naming suggestions based on analysis of high-virality video titles using OpenAI."""
    high_virality_df = df.sort_values(by='virality_score', ascending=False).head(max(1, int(len(df) * 0.1)))

    if high_virality_df.empty:
        return ["No high virality videos found for analysis."]

    titles_to_analyze = high_virality_df['title'].tolist()

    if not titles_to_analyze:
        return ["No titles found in high virality videos for analysis."]

    prompt = f"Analyze these YouTube video titles for common patterns, keywords, and styles that likely contribute to their high virality. Based on this analysis, suggest 5 creative and potentially viral new YouTube video titles for the same topic:\n\n" + "\n".join(titles_to_analyze)

    try:
        openai_api_key = get_secret('OPENAI_API_KEY')  # Use Streamlit secrets or environment variable
        if not openai_api_key:
            st.error("OpenAI API key is not set.")
            return ["OpenAI API key is not set."]
        openai.api_key = openai_api_key

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates YouTube video title suggestions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        suggestions_text = response.choices[0].message.content.strip()
        suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip()]
        return suggestions if suggestions else ["Could not generate naming suggestions."]

    except Exception as e:
        st.error(f"Error generating naming suggestions: {e}")
        return [f"Error generating naming suggestions: {e}"]

def generate_positioning_advice(df):
    """Provides positioning advice based on analysis of high-performing videos using OpenAI."""
    high_virality_df = df.sort_values(by='virality_score', ascending=False).head(max(1, int(len(df) * 0.1)))

    if high_virality_df.empty:
        return ["No high virality videos found for analysis."]

    all_text = " ".join(high_virality_df['text_data'].dropna())
    if not all_text:
         return ["No text data available in high virality videos for analysis."]

    prompt = f"Analyze the following text data from high-virality YouTube videos on a specific topic. Identify recurring themes, keywords, and general strategies evident in the descriptions and tags. Based on this, suggest positioning, messaging, and content strategies for future videos on this topic:\n\n{all_text}"

    try:
        openai_api_key = get_secret('OPENAI_API_KEY')  # Use Streamlit secrets or environment variable
        if not openai_api_key:
            st.error("OpenAI API key is not set.")
            return ["OpenAI API key is not set."]
        openai.api_key = openai_api_key

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides YouTube video positioning advice."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7
        )
        advice_text = response.choices[0].message.content.strip()
        advice = [s.strip() for s in advice_text.split('\n') if s.strip()]
        return advice if advice else ["Could not generate positioning advice."]

    except Exception as e:
        st.error(f"Error generating positioning advice: {e}")
        return [f"Error generating positioning advice: {e}"]

def generate_report(df, naming_suggestions, positioning_advice, detected_topics):
    """Generates a readable report summarizing the video analysis and recommendations in Markdown format."""
    report = "## YouTube Content Analysis Report\n\n"

    report += "### Overview of Analysis\n"
    report += f"- Number of videos analyzed: {len(df)}\n"

    report += "### Video Data Overview\n"
    if not df.empty:
        display_columns = ['title', 'published_at', 'view_count', 'like_count', 'comment_count', 'virality_score', 'cluster_label']
        display_columns = [col for col in display_columns if col in df.columns]

        if display_columns:
            try:
                report += df[display_columns].to_markdown(index=False)
            except ImportError:
                report += "Please install the 'tabulate' library (`pip install tabulate`) to display the video data table.\n"
                for index, row in df.iterrows():
                    report += f"\nTitle: {row.get('title', 'N/A')}\n"
                    report += f"Published: {row.get('published_at', 'N/A')}\n"
                    report += f"Views: {row.get('view_count', 'N/A')}\n"
                    report += f"Likes: {row.get('like_count', 'N/A')}\n"
                    report += f"Comments: {row.get('comment_count', 'N/A')}\n"
                    report += f"Virality Score: {row.get('virality_score', 'N/A'):.2f}\n"
                    report += f"Cluster: {row.get('cluster_label', 'N/A')}\n"
                    report += "-" * 20 + "\n"
        else:
            report += "No relevant columns found to display video data.\n"
    else:
        report += "No video data available.\n"
    report += "\n"

    report += "### Key Findings from Virality Analysis\n"
    if not df.empty and 'virality_score' in df.columns:
        report += f"- Virality Score Range: {df['virality_score'].min():.2f} - {df['virality_score'].max():.2f}\n"
        high_virality_videos = df.sort_values(by='virality_score', ascending=False).head(5)
        if not high_virality_videos.empty:
            report += "- Observations about high-scoring videos (Top 5 titles):\n"
            for index, row in high_virality_videos.iterrows():
                report += f"  - {row.get('title', 'N/A')} (Score: {row.get('virality_score', 0):.2f})\n"
        else:
            report += "- No high-scoring videos found.\n"
    else:
        report += "- No data available for virality analysis.\n"
    report += "\n"

    report += "### Insights from Cluster Analysis\n"
    if 'cluster_label' in df.columns and not df.empty:
        cluster_counts = df['cluster_label'].value_counts()
        report += f"- Number of clusters identified: {len(cluster_counts)}\n"
        report += "- Videos per cluster:\n"
        for cluster, count in cluster_counts.items():
            report += f"  - Cluster {cluster}: {count} videos\n"
        report += "- Characteristics of clusters (based on average metrics - simplified):\n"
        numeric_cols = ['view_count', 'like_count', 'comment_count', 'virality_score']
        available_numeric_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if available_numeric_cols:
             cluster_means = df.groupby('cluster_label')[available_numeric_cols].mean()
             for cluster, means in cluster_means.iterrows():
                report += f"  - Cluster {cluster}: Avg Views={means.get('view_count', 0):.0f}, Avg Likes={means.get('like_count', 0):.0f}, Avg Comments={means.get('comment_count', 0):.0f}, Avg Virality={means.get('virality_score', 0):.2f}\n"
        else:
            report += " - Insufficient numeric data for detailed cluster characteristics.\n"
    else:
        report += "- Cluster analysis not performed or no data available.\n"
    report += "\n"

    report += "### Detected Topic Trends and Key Keywords\n"
    if detected_topics:
        for topic, keywords in detected_topics.items():
            report += f"- **{topic}**: {', '.join(keywords)}\n"
    else:
        report += "- Could not detect topic trends.\n"
    report += "\n"

    report += "### Generated Naming Suggestions\n"
    if naming_suggestions:
        for suggestion in naming_suggestions:
            report += f"- {suggestion}\n"
    else:
        report += "- No specific naming suggestions were generated.\n"
    report += "\n"

    report += "### Generated Positioning Advice\n"
    if positioning_advice:
        for advice_item in positioning_advice:
            report += f"- {advice_item}\n"
    else:
        report += "- No specific positioning advice was generated.\n"
    report += "\n"

    return report

def youtube_analysis_agent(query, n_results):
    """
    Analyzes YouTube videos for a given query and generates a report.
    """
    st.info(f"Searching YouTube for '{query}' and retrieving top {n_results} videos...")
    video_data = search_youtube_videos(query, n_results)

    if not video_data:
        st.warning("No videos found for the given query.")
        return "No videos found for the given query."

    df = pd.DataFrame(video_data)

    st.info("Calculating virality scores...")
    df = calculate_virality_score(df)

    st.info("Performing cluster analysis...")
    df = perform_cluster_analysis(df)

    st.info("Detecting topic trends...")
    detected_topics = detect_topic_trends(df)

    st.info("Generating naming suggestions...")
    naming_suggestions = generate_naming_suggestions(df)

    st.info("Generating positioning advice...")
    positioning_advice = generate_positioning_advice(df)

    st.info("Generating final report...")
    analysis_report = generate_report(df, naming_suggestions, positioning_advice, detected_topics)

    return analysis_report

# --- Streamlit App Layout ---
st.title("YouTube Content Analysis Agent")

# 1. Text input for topic
topic = st.text_input("Enter YouTube Topic:", help="Type the topic you want to analyze on YouTube.")

# 2. Numeric input for number of results
n_results = st.number_input("Number of Results:", min_value=1, value=20, help="Specify how many top videos to analyze.")

# 3. Button to trigger analysis
if st.button("Analyze"):
    if not topic:
        st.warning("Please enter a topic before analyzing.")
    else:
        # Use st.spinner to show a loading message during analysis
        with st.spinner("Analyzing... This might take a few moments."):
            try:
                 report = youtube_analysis_agent(topic, n_results)
                 if report:
                     st.markdown(report)
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
