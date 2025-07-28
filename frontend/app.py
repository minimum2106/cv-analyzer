import streamlit as st
import numpy as np
import fitz
import requests

from cv2 import IMREAD_COLOR, imdecode
from streamlit_image_coordinates import streamlit_image_coordinates
from schemas import JobReqInfo, CVLineInfo
from utils import duplicate_fitz_page

JOB_DOC_MARGIN = 10  # Margin for job description PDF
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


def handle_cv_click():
    coords = st.session_state["cv_click_pos"]

    def map_line_pos_to_img(pos):
        return (
            pos[0] / st.session_state.cv_page_width * coords["width"],
            pos[1] / st.session_state.cv_page_height * coords["height"],
            pos[2] / st.session_state.cv_page_width * coords["width"],
            pos[3] / st.session_state.cv_page_height * coords["height"],
        )

    for line_info in st.session_state.cv_line_info:
        mapped_line_coords = map_line_pos_to_img(line_info.position)

        if (
            mapped_line_coords[0] <= coords["x"] <= mapped_line_coords[2]
            and mapped_line_coords[1] <= coords["y"] <= mapped_line_coords[3]
        ):

            # Update The CV Image
            page = duplicate_fitz_page(st.session_state.cv_org)
            page.add_rect_annot(line_info.position)  # Add annotation to the first block
            pix = page.get_pixmap(dpi=120).tobytes()
            cv2_image = imdecode(
                np.frombuffer(bytearray(pix), dtype=np.uint8), IMREAD_COLOR
            )

            st.session_state.cv_highlighted[0] = cv2_image

            # Update The Job Description Image
            job_page = duplicate_fitz_page(st.session_state.job_org)
            st.session_state.highlighed_job_reqs = []
            for job_req in line_info.connected_job_reqs:
                job_page.add_rect_annot(
                    job_req.position
                )  # Add annotation to the first block
                st.session_state.highlighed_job_reqs.append(job_req)

            job_pix = job_page.get_pixmap(dpi=120).tobytes()
            job_image = imdecode(
                np.frombuffer(bytearray(job_pix), dtype=np.uint8), IMREAD_COLOR
            )
            st.session_state.job_highlighted[0] = job_image

            return


def handle_job_on_click():
    def map_job_pos_to_img(pos):
        return (
            pos[0] / st.session_state.job_page_width * coords["width"],
            pos[1] / st.session_state.job_page_height * coords["height"],
            pos[2] / st.session_state.job_page_width * coords["width"],
            pos[3] / st.session_state.job_page_height * coords["height"],
        )
    
    coords = st.session_state["job_hover_pos"]
    for job_req in st.session_state.highlighed_job_reqs:
        mapped_job_coords = map_job_pos_to_img(job_req.position)

        if (
            mapped_job_coords[0] <= coords["x"] <= mapped_job_coords[2]
            and mapped_job_coords[1] <= coords["y"] <= mapped_job_coords[3]
        ):
            # create a show explanation
            # generate_explanation_on_hover(coords, job_req)
            pass


def generate_cv_line_info(filtered_indices, explanations):
    for idx, (cv_idx, job_idx) in enumerate(filtered_indices):
        cv_info = st.session_state.cv_line_info[cv_idx]
        job_info = st.session_state.job_line_info[job_idx]

        job_info.explanation = explanations[idx]
        cv_info.connected_job_reqs.append(job_info)


def main():
    # Main content
    st.title("📄 CV/Resume Analyzer")
    st.markdown(
        "Upload your CV/Resume and add additional information for AI-powered analysis."
    )

    # Initialize session state
    if "cv_org" not in st.session_state:
        st.session_state.cv_org = None

    if "cv_highlighted" not in st.session_state:
        st.session_state.cv_highlighted = []

    if "cv_line_info" not in st.session_state:
        st.session_state.cv_line_info = []

    if "job_org" not in st.session_state:
        st.session_state.job_org = None

    if "job_highlighted" not in st.session_state:
        st.session_state.job_highlighted = []

    if "highlighed_job_reqs" not in st.session_state:
        st.session_state.highlighed_job_reqs = []

    if "cv_page_width" not in st.session_state:
        st.session_state.cv_page_width = None

    if "cv_page_height" not in st.session_state:
        st.session_state.cv_page_height = None

    if "job_page_width" not in st.session_state:
        st.session_state.job_page_width = None

    if "job_page_height" not in st.session_state:
        st.session_state.job_page_height = None

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
                page = doc.load_page(0)

                st.session_state.cv_page_width = page.mediabox.width
                st.session_state.cv_page_height = page.mediabox.height

                pix = page.get_pixmap(dpi=120).tobytes()
                cv2_image = imdecode(
                    np.frombuffer(bytearray(pix), dtype=np.uint8), IMREAD_COLOR
                )

                st.session_state.cv_org = duplicate_fitz_page(page)
                st.session_state.cv_highlighted.append(cv2_image)

                _ = streamlit_image_coordinates(
                    st.session_state.cv_highlighted[0],
                    key="cv_click_pos",
                    use_column_width=True,
                    on_click=handle_cv_click,
                )

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

            # summarize and get job requirements by LLM
            job_requirements = requests.post(
                f"{BACKEND_URL}/generate_job_requirements",
                json={"job_description": user_text, "model_type": DEFAULT_LLM_SERVICE},
            ).json()["job_requirements"]

            doc = fitz.open()
            doc.new_page()
            page = doc[0]
            rect = fitz.Rect(
                JOB_DOC_MARGIN,
                JOB_DOC_MARGIN,
                page.rect.width - JOB_DOC_MARGIN,
                page.rect.height - JOB_DOC_MARGIN
            )
            page.insert_textbox(rect, user_text, fontsize=12)

            pix = page.get_pixmap(dpi=120).tobytes()
            job_image = imdecode(
                np.frombuffer(bytearray(pix), dtype=np.uint8), IMREAD_COLOR
            )

            st.session_state.job_page_width = page.mediabox.width
            st.session_state.job_page_height = page.mediabox.height

            st.session_state.job_org = duplicate_fitz_page(page)
            st.session_state.job_highlighted.append(job_image)

            # show doc as image
            _ = streamlit_image_coordinates(
                st.session_state.job_highlighted[0],
                key="job_hover_pos",
                use_column_width=True,
                on_click=handle_job_on_click,
            )

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

        with st.spinner(
            f"Analyzing with {DEFAULT_LLM_SERVICE}... This may take a moment."
        ):
            doc = fitz.open(stream=uploaded_cv.getvalue(), filetype="pdf")
            merged_lines = requests.post(
                f"{BACKEND_URL}/extract_text",
                json={"doc": doc},
            ).json()["lines"]

            for line in merged_lines:
                st.session_state.cv_line_info.append(
                    CVLineInfo(
                        position=line.bbox,
                    )
                )

            doc = fitz.open()
            doc.insert_pdf(st.session_state.job_org)
            job_merged_lines = requests.post(
                f"{BACKEND_URL}/extract_text",
                json={"doc": doc},
            ).json()["lines"]

            for line in job_merged_lines:
                st.session_state.job_req_info.append(
                    JobReqInfo(
                        position=line.bbox,
                    )
                )

            similarity_matrix = requests.post(
                f"{BACKEND_URL}/get_similarity_matrix",
                json={
                    "cv_lines": merged_lines.json()["lines"],
                    "job_lines": job_merged_lines.json()["job_lines"],
                },
            ).json()["matrix"]

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

            # add connections between cv line and job requirements
            generate_cv_line_info(filtered_indices, explanations)

    # Footer
    st.markdown("---")
    st.markdown(
        """
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
