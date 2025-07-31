import requests

from cv2 import IMREAD_COLOR, imdecode
import streamlit as st
import numpy as np
import fitz

from streamlit_image_coordinates import streamlit_image_coordinates
from schemas import JobReqInfo, CVLineInfo
from utils import duplicate_fitz_page
from extract_doc import (
    get_lines_with_coords,
    get_bullets_from_doc,
    merge_lines_by_bullets,
    LineWithCoords,
)

JOB_DOC_MARGIN = 10  # Margin for job description PDF
DEFAULT_LLM_SERVICE = "groq"  # Default LLM service
SIMILARITY_THRESHOLD = 0.5  # You may want to tune this
BACKEND_URL = (
    "http://localhost:8000"  # Adjust this if your backend runs on a different port
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


def process_job_description_change():
    job_requirements = requests.post(
        f"{BACKEND_URL}/generate_job_lines",
        json={
            "job_description": st.session_state.user_text,
            "model_type": DEFAULT_LLM_SERVICE,
        },
    ).json()["job_lines"]

    doc = fitz.open()
    doc.new_page()
    page = doc[0]
    rect = fitz.Rect(
        JOB_DOC_MARGIN,
        JOB_DOC_MARGIN,
        page.rect.width - JOB_DOC_MARGIN,
        page.rect.height - JOB_DOC_MARGIN,
    )
    page.insert_textbox(rect, job_requirements, fontsize=12)

    pix = page.get_pixmap(dpi=120).tobytes()
    job_image = imdecode(np.frombuffer(bytearray(pix), dtype=np.uint8), IMREAD_COLOR)

    st.session_state.job_page_width = page.mediabox.width
    st.session_state.job_page_height = page.mediabox.height

    st.session_state.job_org = duplicate_fitz_page(page)
    st.session_state.job_highlighted.append(job_image)

    lines_with_coords, fontsize = get_lines_with_coords(doc)
    bullets = get_bullets_from_doc(doc, fontsize)
    job_merged_lines = merge_lines_by_bullets(lines_with_coords, bullets)

    for line in job_merged_lines:
        st.session_state.job_req_info.append(
            JobReqInfo(
                position=line.bbox,
                text=line.text,
                page=1,
            )
        )


def process_uploaded_cv_change():
    uploaded_cv = st.session_state.uploaded_cv

    if uploaded_cv is None:
        return

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


def generate_cv_line_info(filtered_indices, explanations):
    for idx, (cv_idx, job_idx) in enumerate(filtered_indices):
        cv_info = st.session_state.cv_line_info[cv_idx]
        job_info = st.session_state.job_req_info[job_idx]

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

    if "job_req_info" not in st.session_state:
        st.session_state.job_req_info = []

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
        holder = st.empty()  # Placeholder for the file uploader
        uploaded_cv = holder.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload your CV/Resume in PDF format",
            on_change=process_uploaded_cv_change,
            key="uploaded_cv",
        )

        if uploaded_cv is not None:
            _ = streamlit_image_coordinates(
                st.session_state.cv_highlighted[0],
                key="cv_click_pos",
                use_column_width=True,
                on_click=handle_cv_click,
            )

            holder.empty()  # Clear the placeholder after upload

    # Right column - Text Input
    with col2:
        st.subheader("✏️ Additional Information")
        job_description_holder = st.empty()  # Placeholder for the text area
        user_text = job_description_holder.text_area(
            "Add any additional information:",
            placeholder="Enter job descriptions, specific requirements, career goals, or any other relevant information...",
            height=300,
            key="user_text",  # Bind the text area to session state
            help="Provide context or specific questions for the AI analysis",
            on_change=process_job_description_change,
        )

        # # Character count
        if user_text:
            _ = streamlit_image_coordinates(
                st.session_state.job_highlighted[0],
                key="job_hover_pos",
                use_column_width=True,
                on_click=handle_job_on_click,
            )


    # Analysis section
    st.markdown("---")

    # Analysis button
    _, col_btn2, _ = st.columns([1, 2, 1])
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
            lines_with_coords, fontsize = get_lines_with_coords(doc)
            bullets = get_bullets_from_doc(doc, fontsize)
            cv_merged_lines = merge_lines_by_bullets(lines_with_coords, bullets)

            for line in cv_merged_lines:
                st.session_state.cv_line_info.append(
                    CVLineInfo(
                        position=line.bbox,
                        text=line.text,
                        page=1,
                    )
                )

            doc = fitz.open()
            doc.new_page()
            doc[0].insert_textbox(
                fitz.Rect(
                    JOB_DOC_MARGIN,
                    JOB_DOC_MARGIN,
                    doc[0].rect.width - JOB_DOC_MARGIN,
                    doc[0].rect.height - JOB_DOC_MARGIN,
                ),
                user_text,
                fontsize=fontsize,
            )
            lines_with_coords, fontsize = get_lines_with_coords(doc)
            bullets = get_bullets_from_doc(doc, fontsize)
            job_merged_lines = merge_lines_by_bullets(lines_with_coords, bullets)

            for line in job_merged_lines:
                st.session_state.job_req_info.append(
                    JobReqInfo(
                        position=line.bbox,
                        text=line.text,
                        page=1,
                    )
                )

            similarity_matrix = requests.post(
                f"{BACKEND_URL}/get_similarity_matrix",
                json={
                    "cv_lines": [line.text for line in cv_merged_lines],
                    "job_lines": [line.text for line in job_merged_lines],
                },
            ).json()["matrix"]

            filtered_indices = np.argwhere(
                np.array(similarity_matrix) > SIMILARITY_THRESHOLD
            )
            print("Filtered Indices:", len(filtered_indices))
            response = requests.post(
                f"{BACKEND_URL}/explain_match",
                json={
                    "model_type": DEFAULT_LLM_SERVICE,
                    "cv_lines": [line.text for line in cv_merged_lines],
                    "job_lines": [line.text for line in job_merged_lines],
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
