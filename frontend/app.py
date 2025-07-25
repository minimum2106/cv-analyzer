import streamlit as st
import numpy as np
import fitz
import requests
from streamlit_image_coordinates import streamlit_image_coordinates


DEFAULT_LLM_SERVICE = "openai"  # Default LLM service
THRESHOLD = 0.7  # You may want to tune this
BACKEND_URL = (
    "http://localhost:5173"  # Adjust this if your backend runs on a different port
)

# Configure page
st.set_page_config(
    page_title="CV Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

valid_coords = []
CV_IMAGES = []

def duplicate_fitz_page(page):
    """
    Create a duplicate of a PyMuPDF page object.
    Returns a new page object with the same content.
    """
    # Export the page as a PDF bytes
    pdf_bytes = page.parent.write([page.number])
    # Open a new document from the single-page PDF
    new_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Return the first (and only) page of the new document
    return new_doc.load_page(0)


def handle_cv_click():
    coords = st.session_state["cv_click_pos"]
    for valid_coord in valid_coords:
        if (
            valid_coord["x1"] <= coords["x"] <= valid_coord["x2"]
            and valid_coord["y1"] <= coords["y"] <= valid_coord["y2"]
        ):
            st.write(f"Clicked on: {valid_coord['text']}")
            page = duplicate_fitz_page(CV_IMAGES[0])
            page.add_rect_annot(
                    valid_coord
            )  # Add annotation to the first block
            st.session_state.cv_idx += 1
            if st.session_state.cv_idx >= len(CV_IMAGES):
                st.session_state.cv_idx = 0

            return


def main():
    # Initialize session state
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False

    if 'cv_idx' not in st.session_state:
        st.session_state.cv_idx = 0

    if 'cv_imgs' not in st.session.state:
        st.session_state.cv_imgs = []

    # Main content
    st.title("📄 CV/Resume Analyzer")
    st.markdown(
        "Upload your CV/Resume and add additional information for AI-powered analysis."
    )

    # Create two columns for the main content
    col1, col2 = st.columns(2)

    # Left column - PDF Upload
    with col1:
        st.subheader("📁 Upload PDF")
        uploaded_cv = st.file_uploader(
            "Choose a PDF file", type="pdf", help="Upload your CV/Resume in PDF format"
        )

        if uploaded_cv is not None:
            with fitz.open(stream=uploaded_cv.getvalue()) as doc:
                page_number = st.sidebar.number_input(
                    "Page number",
                    min_value=1,
                    max_value=doc.page_count,
                    value=1,
                    step=1,
                )
                page = doc.load_page(page_number - 1)
                pix = page.get_pixmap(dpi=120).tobytes()
                # CV_IMAGES.append(pix)
                # _ = streamlit_image_coordinates(
                #     CV_IMAGES[st.session_state.cv_idx],
                #     key="cv_click_pos",
                #     use_column_width=True,
                #     on_click=handle_cv_click,
                # )
                st.image(pix, use_container_width=True)

    # Right column - Text Input
    with col2:
        st.subheader("✏️ Additional Information")
        user_text = st.text_area(
            "Add any additional information:",
            placeholder="Enter job descriptions, specific requirements, career goals, or any other relevant information...",
            height=300,
            help="Provide context or specific questions for the AI analysis",
        )

        # Character count
        if user_text:
            st.caption(f"Characters: {len(user_text)}")

    # Analysis section
    st.markdown("---")

    # Analysis button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button(
            "🔍 Analyze CV/Resume",
            type="primary",
            use_container_width=True,
            disabled=not (uploaded_cv and user_text),
        )

    # Show requirements if button is disabled
    if not (uploaded_cv and user_text):
        st.warning(
            "⚠️ Please upload a PDF file AND add additional information to enable analysis."
        )

    if analyze_button and uploaded_cv and user_text:
        st.subheader("🎯 AI Analysis Results")

        with st.spinner(f"Analyzing with {DEFAULT_LLM_SERVICE}... This may take a moment."):
            doc = fitz.open(stream=uploaded_cv.read(), filetype="pdf")
            merged_lines = requests.post(
                f"{BACKEND_URL}/extract_cv_text",
                json={"doc": doc},
            )

            job_merged_lines = requests.post(
                f"{BACKEND_URL}/extract_job_lines",
                json={"job_description": user_text, "model_type": DEFAULT_LLM_SERVICE},
            )

            similarity_matrix = requests.post(
                f"{BACKEND_URL}/get_similarity_matrix",
                json={
                    "cv_lines": merged_lines.json()["lines"],
                    "job_lines": job_merged_lines.json()["job_lines"],
                },
            )

            filtered_indices = np.argwhere(similarity_matrix > THRESHOLD)
            response = requests.post(
                f"{BACKEND_URL}/explain_matches",
                json={
                    "model_type": DEFAULT_LLM_SERVICE,
                    "cv_lines": merged_lines.json()["lines"],
                    "job_lines": job_merged_lines.json()["job_lines"],
                    "filtered_indices": filtered_indices.tolist(),
                },
            )
            explanations = response.json()["explanations"]
            
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
