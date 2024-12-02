import streamlit as st
import requests
from PIL import Image
import io
import logging
import yaml
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load configuration
with open('configs/config.yml', 'r') as f:
    config = yaml.safe_load(f)
    
API_URL = config['fastapi_api_search_url']

def process_image(uploaded_file) -> tuple[bool, str]:
    """
    Summary: Process uploaded image through FastAPI endpoint
    
    Args:
        uploaded_file (UploadFile): The uploaded image file
        
    Returns:
        tuple[bool, str]: Success status and error message if any
    """
    try:
        files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
        response = requests.post(API_URL, files=files)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return False, f"Error connecting to API: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False, f"Unexpected error: {str(e)}"

def display_results(matches: list) -> None:
    """
    Summary: Display search results in Streamlit interface
    
    Args:
        matches (list): List of matching logos with similarity scores
    """
    if not matches:
        st.warning("No similar logos found.")
        return

    # Display unique companies
    st.subheader("Identified Companies")
    unique_companies = {match["company_name"] for match in matches}
    for company in unique_companies:
        st.write(f"• {company}")

    # Display detailed results
    st.subheader("Similar Images")
    for match in matches:
        with st.expander(f"{match['company_name']} - Similarity: {match['similarity_score']:.2f}"):
            st.write(f"Image Name: {match['image_name']}")
            st.write(f"Similarity Score: {match['similarity_score']:.4f}")

def main():
    """
    Summary: Main Streamlit application
    """
    st.set_page_config(
        page_title="Logo Identification Service",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Logo Identification Service")
    st.write("Upload an image to identify logos and find similar images.")

    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing a logo"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        if st.button("🔎 Identify Logo", type="primary"):
            with st.spinner("Processing image..."):
                success, result = process_image(uploaded_file)
                
                if success:
                    with col2:
                        display_results(result["matches"])
                else:
                    st.error(result)

if __name__ == "__main__":
    main()
