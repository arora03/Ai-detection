import streamlit as st
import joblib
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import PyPDF2
import io
# --- NEW IMPORT ---
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="🛡️ AIShield — AI Content Detector",
    page_icon="🛡️",
    layout="wide"
)

# ----------------------------------------------------------------------
# NEW: Define CSS for Light (Yellow) and Dark Themes
# ----------------------------------------------------------------------
# (This is the same 200+ line CSS block as before)
# (It's hidden here for brevity, but it's in the full code below)
LIGHT_CSS = """
<style>
    :root {
        --primary-color: #ff4b4b; /* Red */
        --app-bg: #FFFFF0; /* Light Pastel Yellow */
        --sidebar-bg: #FDF5E6; /* Old Lace */
        --card-bg: #FFFFFF; /* White */
        --text-color: #333333; /* Dark text */
        --subtle-text: #555555;
        --track-color: #000000; /* Black slider track */
        
        --ai-red-bg: #FFF0F0;
        --ai-red-border: #FF8A8A;
        --human-green-bg: #F0FFF0;
        --human-green-border: #8AFF8A;
        
        --info-bg: #E6F3FF; /* Light blue info box */
        --info-text: #004C99;
    }
    
    /* Apply base styles */
    body, .stApp {
        background-color: var(--app-bg) !important;
        color: var(--text-color) !important;
    }
    h1, h2, h3, h4, h5, h6, .stTabs [data-testid="stTab"] {
        color: var(--text-color) !important;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] p {
        color: var(--text-color) !important;
    }
    [data-testid="stSidebar"] [data-testid^="stVerticalBlock"] {
        background-color: transparent !important; /* Fix white boxes */
    }

    /* --- Metrics --- */
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #28a745 !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #28a745 !important; }
    [data-testid="stMain"] [data-testid="stMetricValue"] { color: var(--primary-color) !important; }
    [data-testid="stMetricHelpIcon"] { color: var(--subtle-text) !important; }

    /* --- Slider --- */
    [data-testid="stSlider"] div[data-testid="stTickBar"] div {
        background-color: var(--track-color) !important;
    }
    [data-testid="stSlider"] span[role="slider"] {
        background-color: var(--primary-color) !important;
    }

    /* --- Analyze Button --- */
    [data-testid="stButton"] button[kind="primary"] {
        background-color: var(--primary-color) !important;
        border: none !important;
    }
    [data-testid="stButton"] button[kind="primary"] div,
    [data-testid="stButton"] button[kind="primary"] p {
        color: white !important;
        background-color: transparent !important;
    }

    /* --- Text Input & File Uploader --- */
    [data-testid="stTextArea"] textarea,
    [data-testid="stFileUploader"] {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid #e0e0e0 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: var(--card-bg) !important;
        border: 2px dashed #AAAAAA !important;
    }
    [data-testid="stFileUploader"] label, [data-testid="stFile"] > div {
        color: var(--text-color) !important;
    }
    [data-testid="stFileUploader"] small { color: var(--subtle-text) !important; }
    [data-testid="stBrowseButton"] button {
        background-color: #F0F0F0 !important;
        color: #333333 !important;
        border: 1px solid #AAAAAA !important;
    }
    [data-testid="stBrowseButton"] button p {
        color: #333333 !important; 
    }

    /* --- Info Box --- */
    [data-testid="stInfo"], [data-testid="stInfo"] div[role="alert"] {
        background-color: var(--info-bg) !important;
        color: var(--info-text) !important;
    }

    /* --- Expander "Buttons" --- */
    div[data-testid="stExpander"] summary {
        background-color: var(--card-bg);
        border: 1px solid #e6e6e0;
        color: var(--text-color) !important;
    }

    /* --- Highlight Cards --- */
    .ai-card {
        background-color: var(--ai-red-bg);
        border-left: 6px solid var(--ai-red-border);
        color: var(--text-color) !important;
        padding: 12px;
        border-radius: 8px;
    }
    .human-card {
        background-color: var(--human-green-bg);
        border-left: 6px solid var(--human-green-border);
        color: var(--text-color) !important;
        padding: 12px;
        border-radius: 8px;
    }
</style>
"""

DARK_CSS = """
<style>
    :root {
        --primary-color: #ff4b4b; /* Red */
        --app-bg: #0E1117; /* Streamlit's dark background */
        --sidebar-bg: #1F222B; /* A lighter dark for cards */
        --card-bg: #1F222B;
        --text-color: #FAFAFA; /* Light text */
        --subtle-text: #AAAAAA;
        --track-color: #555555; /* Lighter slider track */

        --ai-red-bg: #402020;
        --ai-red-border: #ff4b4b;
        --human-green-bg: #204020;
        --human-green-border: #28a745;
        
        --info-bg: #0E2E4A; /* Dark blue info box */
        --info-text: #E6F3FF;
    }
    
    /* Apply base styles */
    body, .stApp {
        background-color: var(--app-bg) !important;
        color: var(--text-color) !important;
    }
    h1, h2, h3, h4, h5, h6, .stTabs [data-testid="stTab"] {
        color: var(--text-color) !important;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid #333333;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] p {
        color: var(--text-color) !important;
    }
    [data-testid="stSidebar"] [data-testid^="stVerticalBlock"] {
        background-color: transparent !important; /* Fix white boxes */
    }

    /* --- Metrics --- */
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #28a745 !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #28a745 !important; }
    [data-testid="stMain"] [data-testid="stMetricValue"] { color: var(--primary-color) !important; }
    [data-testid="stMetricHelpIcon"] { color: var(--subtle-text) !important; }

    /* --- Slider --- */
    [data-testid="stSlider"] div[data-testid="stTickBar"] div {
        background-color: var(--track-color) !important;
    }
    [data-testid="stSlider"] span[role="slider"] {
        background-color: var(--primary-color) !important;
    }

    /* --- Analyze Button --- */
    [data-testid="stButton"] button[kind="primary"] {
        background-color: var(--primary-color) !important;
        border: none !important;
    }
    [data-testid="stButton"] button[kind="primary"] div,
    [data-testid="stButton"] button[kind="primary"] p {
        color: white !important;
        background-color: transparent !important;
    }

    /* --- Text Input & File Uploader --- */
    [data-testid="stTextArea"] textarea,
    [data-testid="stFileUploader"] {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid #333333 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: var(--card-bg) !important;
        border: 2px dashed #555555 !important;
    }
    [data-testid="stFileUploader"] label, [data-testid="stFile"] > div {
        color: var(--text-color) !important;
    }
    [data-testid="stFileUploader"] small { color: var(--subtle-text) !important; }
    [data-testid="stBrowseButton"] button {
        background-color: #333333 !important;
        color: var(--text-color) !important;
        border: 1px solid #555555 !important;
    }
    [data-testid="stBrowseButton"] button p {
        color: var(--text-color) !important; 
    }

    /* --- Info Box --- */
    [data-testid="stInfo"], [data-testid="stInfo"] div[role="alert"] {
        background-color: var(--info-bg) !important;
        color: var(--info-text) !important;
    }

    /* --- Expander "Buttons" --- */
    div[data-testid="stExpander"] summary {
        background-color: var(--card-bg);
        border: 1px solid #333333;
        color: var(--text-color) !important;
    }

    /* --- Highlight Cards --- */
    .ai-card {
        background-color: var(--ai-red-bg);
        border-left: 6px solid var(--ai-red-border);
        color: var(--text-color) !important;
        padding: 12px;
        border-radius: 8px;
    }
    .human-card {
        background-color: var(--human-green-bg);
        border-left: 6px solid var(--human-green-border);
        color: var(--text-color) !important;
        padding: 12px;
        border-radius: 8px;
    }
</style>
"""

# ----------------------------------------------------------------------
# Apply Theme based on Toggle
# ----------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False 

if st.session_state.dark_mode:
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Load Model (Cached for speed)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/trained_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------------------------------------------------
# Helper Function (To read PDF files)
# ----------------------------------------------------------------------
def extract_text_from_pdf(file_like_object):
    try:
        pdf_reader = PyPDF2.PdfReader(file_like_object)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h1>🛡️ AIShield</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 20px;'>AI Content & Plagiarism Detector</p>", unsafe_allow_html=True)

    # --- Model Accuracy Display (Will be Green) ---
    st.metric(
        label="Model Accuracy",
        value="92.5%", # <-- HARDCODE YOUR *NEW* ACCURACY HERE
        help="This is the accuracy of our AI/Human classifier on its test dataset."
    )
    
    st.markdown("---")

    # --- Threshold Slider ---
    st.subheader("Analysis Controls")
    threshold = st.slider(
        "AI Probability Threshold:",
        min_value=0,
        max_value=100,
        value=50, # Default to 50%
        format="%d%%",
        help="Sentences with an AI score *higher* than this will be flagged as AI."
    )
    
    st.markdown("---")
    
    # --- Dark Mode Toggle ---
    st.toggle("Toggle Dark Mode", key="dark_mode")
    
    st.markdown("---")
    
    analyze_button = st.button("Analyze Text", use_container_width=True, type="primary")

# ----------------------------------------------------------------------
# Main App UI
# ----------------------------------------------------------------------

# --- NEW: Define the two main tabs ---
ai_tab, plag_tab = st.tabs(["🤖 AI Content Detector", "📝 Plagiarism Checker"])

# --- TAB 1: AI CONTENT DETECTOR ---
with ai_tab:
    left_col, right_col = st.columns(2)

    # --- Input Column ---
    with left_col:
        st.header("Your Input")
        tab1, tab2 = st.tabs(["Paste Text", "Upload PDF"])

        with tab1:
            ai_text_input = st.text_area("Paste your text here to analyze...", height=400, placeholder="Start typing or paste your content...", key="ai_input")
        
        with tab2:
            ai_uploaded_file = st.file_uploader("Or, upload a PDF document", type=["pdf"], key="ai_upload")

    # --- Results Column ---
    with right_col:
        st.header("Analysis Results")
        
        if analyze_button:
            final_text = ""
            if ai_uploaded_file:
                final_text = extract_text_from_pdf(ai_uploaded_file)
            elif ai_text_input.strip() != "":
                final_text = ai_text_input
            else:
                st.warning("Please paste text or upload a PDF file to analyze.")

            if final_text:
                sentences = sent_tokenize(final_text)
                
                if not sentences:
                    st.error("Could not find any sentences to analyze.")
                else:
                    probs = []
                    for sentence in sentences:
                        vec = vectorizer.transform([sentence])
                        prob_ai = model.predict_proba(vec)[0][1] 
                        probs.append(prob_ai)
                    
                    avg_prob = np.mean(probs) * 100

                    st.metric(
                        label="Overall AI Probability Score",
                        value=f"{avg_prob:.1f}%",
                        help="This is the average probability that a sentence in this text is AI-generated."
                    )
                    st.markdown("---")

                    html_results_ai = []
                    html_results_human = []
                    
                    for i, sentence in enumerate(sentences):
                        prob_percent = probs[i] * 100
                        if prob_percent >= threshold:
                            html_results_ai.append(
                                f"<div class='ai-card'>"
                                f"{sentence}<br><small><b>(AI Score: {prob_percent:.1f}%)</b></small>"
                                f"</div>"
                            )
                        else:
                            html_results_human.append(
                                f"<div class='human-card'>"
                                f"{sentence}<br><small><b>(AI Score: {prob_percent:.1f}%)</b></small>"
                                f"</div>"
                            )
                    
                    ai_expander_label = f"🔴 Flagged as AI (>{threshold}%) — {len(html_results_ai)} sentences"
                    with st.expander(ai_expander_label, expanded=True):
                        if not html_results_ai:
                            st.success(f"Good news! No sentences were found with an AI probability above {threshold}%.")
                        else:
                            st.markdown("".join(html_results_ai), unsafe_allow_html=True)

                    human_expander_label = f"🟢 Flagged as Human (<{threshold}%) — {len(html_results_human)} sentences"
                    with st.expander(human_expander_label, expanded=False):
                        if not html_results_human:
                            st.write("No sentences were found below the threshold.")
                        else:
                            st.markdown("".join(html_results_human), unsafe_allow_html=True)

        else:
            st.info("Paste text or upload a PDF, then click 'Analyze Text' in the sidebar to see the results.")

# --- TAB 2: PLAGIARISM CHECKER (NEW FEATURE) ---
with plag_tab:
    st.header("Check for Plagiarism Between Two Texts")
    
    plag_col1, plag_col2 = st.columns(2)
    
    with plag_col1:
        st.subheader("Original Source Text")
        source_text = st.text_area("Paste the original text here.", height=300, key="plag_source")
    
    with plag_col2:
        st.subheader("Suspect Text to Check")
        suspect_text = st.text_area("Paste the text you want to check against the source.", height=300, key="plag_suspect")
    
    if analyze_button:
        if source_text.strip() != "" and suspect_text.strip() != "":
            # Use the same vectorizer from our AI model
            try:
                vectors = vectorizer.transform([source_text, suspect_text])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])
                
                # The result is a 2D array, so we get the single value
                similarity_score = similarity[0][0] * 100
                
                st.metric(
                    label="Text Similarity Score",
                    value=f"{similarity_score:.1f}%",
                    help="This is the percentage of similarity between the 'Source Text' and the 'Suspect Text' based on their shared vocabulary."
                )
                
                # Add a simple progress bar
                st.progress(int(similarity_score) / 100)
                
                if similarity_score > 75:
                    st.error("High Similarity Detected: This text is very similar to the source text.")
                elif similarity_score > 40:
                    st.warning("Moderate Similarity Detected: This text shares some significant parts with the source text.")
                else:
                    st.success("Low Similarity: This text appears to be substantially different from the source text.")

            except Exception as e:
                st.error(f"Could not perform similarity check. Error: {e}")
        elif analyze_button:
            st.warning("Please provide both a 'Source Text' and a 'Suspect Text' to run the plagiarism check.")
    else:
        st.info("Paste your source text and suspect text, then click 'Analyze Text' in the sidebar to check for similarity.")