import streamlit as st
import cv2
import tempfile
import numpy as np
from gait_processor import GaitProcessor
import matplotlib.pyplot as plt
import pandas as pd
from database import save_database
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

def create_real_time_plot():
    """Create and return a real-time plotly figure."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], name='Left Knee', mode='lines'))
    fig.add_trace(go.Scatter(x=[], y=[], name='Right Knee', mode='lines'))
    fig.add_trace(go.Scatter(x=[], y=[], name='Left Hip', mode='lines'))
    fig.add_trace(go.Scatter(x=[], y=[], name='Right Hip', mode='lines'))
    
    fig.update_layout(
        title='Real-time Joint Angles',
        xaxis_title='Frame',
        yaxis_title='Angle (degrees)',
        height=400
    )
    return fig

def main():
    st.title("Criminal Gait Analysis System")
    
    # Load database
    try:
        df = pd.read_csv('criminal_gait_database.csv')
    except FileNotFoundError:
        df = save_database()
    
    # Sidebar for analysis options
    st.sidebar.header("Analysis Options")
    analysis_mode = st.sidebar.radio(
        "Select Mode",
        ["Live Analysis", "Database Search", "Pattern Matching"]
    )
    
    if analysis_mode == "Live Analysis":
        run_live_analysis()
    elif analysis_mode == "Database Search":
        run_database_search(df)
    else:
        run_pattern_matching(df)

def run_live_analysis():
    st.header("Live Gait Analysis")
    source = st.radio("Select Source", ["Upload Video", "CCTV Feed", "Webcam"])
    
    # Initialize processor and plots
    processor = GaitProcessor()
    metrics_plot = create_real_time_plot()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        frame_placeholder = st.empty()
    with col2:
        metrics_placeholder = st.empty()
        match_placeholder = st.empty()
    
    if source == "Upload Video":
        video_file = st.file_uploader("Upload surveillance footage", type=['mp4', 'avi', 'mov'])
        if video_file:
            process_surveillance(video_file, processor, frame_placeholder, 
                              metrics_placeholder, match_placeholder)

def process_surveillance(video_file, processor, frame_placeholder, 
                       metrics_placeholder, match_placeholder):
    """Process surveillance footage with criminal gait analysis."""
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        annotated_frame, metrics = processor.process_frame(frame)
        
        # Update displays
        frame_placeholder.image(annotated_frame, channels="BGR")
        
        if metrics and metrics.gait_pattern:
            metrics_placeholder.write(f"""
                ### Match Found
                Confidence: {metrics.confidence_score:.2f}
                Gait Pattern: {metrics.gait_pattern}
                Symmetry: {metrics.gait_symmetry:.2f}
                Cadence: {metrics.cadence:.1f} steps/min
            """)
            
            # Show matching suspect info if confidence is high
            if metrics.confidence_score > 0.8:
                suspect_info = get_suspect_info(metrics.gait_pattern)
                match_placeholder.warning(f"""
                    🚨 **Potential Match**
                    Suspect ID: {suspect_info['suspect_id']}
                    Criminal History: {suspect_info['criminal_history']}
                    Last Seen: {suspect_info['last_seen']}
                """)
    
    cap.release()

def get_suspect_info(suspect_id):
    """Retrieve suspect information from database."""
    df = pd.read_csv('criminal_gait_database.csv')
    suspect = df[df['suspect_id'] == suspect_id].iloc[0]
    return suspect.to_dict()

def run_database_search(df):
    """Handle database search functionality."""
    st.header("Criminal Database Search")
    
    # Search filters
    col1, col2 = st.columns(2)
    
    with col1:
        search_name = st.text_input("Search by Name")
        age_range = st.slider("Age Range", 18, 65, (25, 45))
        
    with col2:
        gait_pattern = st.selectbox(
            "Gait Pattern",
            ['All'] + list(df['gait_pattern'].unique())
        )
        criminal_history = st.selectbox(
            "Criminal History",
            ['All'] + list(df['criminal_history'].unique())
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_name:
        filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False)]
    
    filtered_df = filtered_df[
        (filtered_df['age'] >= age_range[0]) &
        (filtered_df['age'] <= age_range[1])
    ]
    
    if gait_pattern != 'All':
        filtered_df = filtered_df[filtered_df['gait_pattern'] == gait_pattern]
        
    if criminal_history != 'All':
        filtered_df = filtered_df[filtered_df['criminal_history'] == criminal_history]
    
    # Display results
    st.subheader(f"Search Results ({len(filtered_df)} records)")
    st.dataframe(filtered_df)
    
    # Detailed view of selected suspect
    if st.checkbox("Show Detailed Analysis"):
        selected_suspect = st.selectbox(
            "Select Suspect for Detailed Analysis",
            filtered_df['suspect_id'].tolist()
        )
        
        suspect_data = filtered_df[filtered_df['suspect_id'] == selected_suspect].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Suspect Information")
            st.write(f"Name: {suspect_data['name']}")
            st.write(f"Age: {suspect_data['age']}")
            st.write(f"Gender: {suspect_data['gender']}")
            st.write(f"Height: {suspect_data['height']} cm")
            st.write(f"Weight: {suspect_data['weight']} kg")
            
        with col2:
            st.write("### Gait Analysis")
            st.write(f"Gait Pattern: {suspect_data['gait_pattern']}")
            st.write(f"Criminal History: {suspect_data['criminal_history']}")
            st.write(f"Last Seen: {suspect_data['last_seen']}")
            
            # Display gait signature details
            gait_signature = eval(suspect_data['gait_signature'])
            st.write("### Gait Signature")
            st.write(f"Stride Length: {gait_signature['stride_length_avg']:.1f} cm")
            st.write(f"Step Width: {gait_signature['step_width_avg']:.1f} cm")
            st.write(f"Cadence: {gait_signature['cadence']:.1f} steps/min")
            st.write(f"Gait Symmetry: {gait_signature['gait_symmetry']:.2f}")
            st.write(f"Distinctive Features: {gait_signature['distinctive_features']}")

def run_pattern_matching(df):
    """Handle pattern matching functionality."""
    st.header("Gait Pattern Matching")
    
    # Input for new gait measurements
    st.subheader("Enter Gait Measurements")
    col1, col2 = st.columns(2)
    
    with col1:
        stride_length = st.number_input("Stride Length (cm)", 40.0, 100.0, 75.0)
        step_width = st.number_input("Step Width (cm)", 5.0, 30.0, 15.0)
        
    with col2:
        cadence = st.number_input("Cadence (steps/min)", 60.0, 150.0, 110.0)
        gait_symmetry = st.slider("Gait Symmetry", 0.0, 1.0, 0.9)
    
    if st.button("Find Matches"):
        # Create feature vector
        input_features = np.array([stride_length, step_width, cadence, gait_symmetry])
        
        # Calculate similarities
        similarities = []
        for _, row in df.iterrows():
            gait_signature = eval(row['gait_signature'])
            known_features = np.array([
                gait_signature['stride_length_avg'],
                gait_signature['step_width_avg'],
                gait_signature['cadence'],
                gait_signature['gait_symmetry']
            ])
            
            similarity = cosine_similarity(
                input_features.reshape(1, -1),
                known_features.reshape(1, -1)
            )[0][0]
            
            similarities.append({
                'suspect_id': row['suspect_id'],
                'name': row['name'],
                'similarity': similarity,
                'criminal_history': row['criminal_history']
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Display top matches
        st.subheader("Top Matches")
        for i, match in enumerate(similarities[:5]):
            st.write(f"""
                Match {i+1}: {match['name']} (ID: {match['suspect_id']})
                - Similarity: {match['similarity']:.2%}
                - Criminal History: {match['criminal_history']}
                ---
            """)

if __name__ == "__main__":
    main() 