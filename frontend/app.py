import streamlit as st
import pymupdf
import openai
import anthropic
from groq import Groq
import numpy as np
import fitz


# Configure page
st.set_page_config(
    page_title="CV Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def extract_pdf_text(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = pymupdf.open(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return ""


def get_llm_response(
    model_type: str, temperature: float, pdf_text: str, user_text: str
) -> str:
    """Get response from selected LLM model"""

    prompt = f"""
    Please analyze the following CV/Resume and additional information:
    
    CV/Resume Content:
    {pdf_text}
    
    Additional Information:
    {user_text}
    
    Please provide a comprehensive analysis including:
    1. Key skills and qualifications
    2. Experience summary
    3. Strengths and areas for improvement
    4. Recommendations for career development
    """

    try:
        if model_type == "OpenAI":
            if not st.secrets.get("OPENAI_API_KEY"):
                return "OpenAI API key not configured. Please add it to your Streamlit secrets."

            client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1500,
            )
            return response.choices[0].message.content

        elif model_type == "Claude":
            if not st.secrets.get("ANTHROPIC_API_KEY"):
                return "Claude API key not configured. Please add it to your Streamlit secrets."

            client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif model_type == "Groq":
            if not st.secrets.get("GROQ_API_KEY"):
                return "Groq API key not configured. Please add it to your Streamlit secrets."

            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1500,
            )
            return response.choices[0].message.content

    except Exception as e:
        return f"Error getting response from {model_type}: {str(e)}"


def bbox_vertically_close(line_bbox, bullet_rect, tolerance=5):
    # Check if the vertical center of the bullet is close to the line's vertical span
    _, line_y0, _, line_y1 = line_bbox
    _, bullet_y0, _, bullet_y1 = bullet_rect
    return (bullet_y0 >= line_y0 - tolerance) and (bullet_y1 <= line_y1 + tolerance)


def merge_lines_by_bullets(
    lines_with_coords, bullets, tolerance=5, line_gap_factor=1.1
):
    """
    Merge lines into bullet groups based on proximity to bullet graphics.
    Stops merging when the next bullet is reached, when the vertical gap between lines is unusually large,
    or when the font size changes.
    Only merges lines whose top y (y0) is equal or after the y of the bullet point.
    """
    merged = []
    used = set()
    # Group bullets by page and sort by vertical position (top to bottom)
    bullets_by_page = {}
    for bullet in bullets:
        page = bullet.get("page", 1)
        bullets_by_page.setdefault(page, []).append(bullet)
    for page, page_bullets in bullets_by_page.items():
        # Sort bullets by their top y coordinate
        page_bullets = sorted(page_bullets, key=lambda b: b["rect"].y0)
        # Get all lines on this page, sorted by their top y
        page_lines = [
            (idx, line)
            for idx, line in enumerate(lines_with_coords)
            if line["page"] == page
        ]
        page_lines = sorted(page_lines, key=lambda x: x[1]["bbox"][1])

        # Estimate normal line gap for this page
        line_gaps = []
        for i in range(1, len(page_lines)):
            prev_y1 = page_lines[i - 1][1]["bbox"][3]
            curr_y0 = page_lines[i][1]["bbox"][1]
            line_gaps.append(curr_y0 - prev_y1)

        normal_gap = (
            np.median([gap for gap in line_gaps if gap > 0]) if line_gaps else 0
        )

        for i, bullet in enumerate(page_bullets):
            bullet_rect = bullet["rect"]
            bullet_y0 = bullet_rect.y0
            # Determine the y0 of the next bullet, or end of page
            if i + 1 < len(page_bullets):
                next_bullet_y0 = page_bullets[i + 1]["rect"].y0
            else:
                next_bullet_y0 = float("inf")
            group_lines = []
            group_bbox = None
            last_line_y1 = None
            bullet_fontsize = None  # <-- change here
            for idx, line in page_lines:
                if idx in used:
                    continue
                line_y0 = line["bbox"][1]
                line_y1 = line["bbox"][3]
                # Only include lines whose top y is equal or after the bullet y
                if (line_y0 >= bullet_y0 - tolerance) and (
                    line_y0 < next_bullet_y0 - tolerance
                ):
                    # If this is the first line in the group, set bullet_fontsize
                    if bullet_fontsize is None:
                        bullet_fontsize = line.get("fontsize")
                    # If this is not the first line in the group, check the gap and font size
                    if last_line_y1 is not None and normal_gap > 0:
                        gap = line_y0 - last_line_y1
                        if gap > line_gap_factor * normal_gap:
                            # Large gap, stop merging for this bullet
                            break
                    if (
                        bullet_fontsize is not None
                        and "fontsize" in line
                        and line["fontsize"] != bullet_fontsize
                    ):
                        # Font size changed, stop merging for this bullet
                        break
                    group_lines.append(line)
                    used.add(idx)
                    x0, y0, x1, y1 = line["bbox"]
                    if group_bbox is None:
                        group_bbox = [x0, y0, x1, y1]
                    else:
                        group_bbox[0] = min(group_bbox[0], x0)
                        group_bbox[1] = min(group_bbox[1], y0)
                        group_bbox[2] = max(group_bbox[2], x1)
                        group_bbox[3] = max(group_bbox[3], y1)
                    last_line_y1 = line_y1
                else:
                    continue
            if group_lines:
                merged_text = " ".join([l["text"] for l in group_lines])
                merged.append(
                    {"text": merged_text, "bbox": tuple(group_bbox), "page": page}
                )
    # Add lines not assigned to any bullet
    for idx, line in enumerate(lines_with_coords):
        if idx not in used:
            merged.append(line)

    return merged


def main():
    # Initialize session state
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False

    # Sidebar
    with st.sidebar:
        st.title("🤖 AI Settings")

        # Model selection
        st.subheader("Select LLM Model")
        model_type = st.selectbox(
            "Choose your AI model:",
            ["OpenAI", "Claude", "Groq"],
            help="Select which AI model to use for analysis",
        )

        # Temperature slider
        st.subheader("Model Parameters")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness: 0.0 = focused, 2.0 = creative",
        )

        # Model info
        st.info(f"**Selected Model:** {model_type}\n**Temperature:** {temperature}")

        # Sidebar collapse button
        if st.button("🔄 Toggle Sidebar"):
            st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
            st.rerun()

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
        uploaded_file = st.file_uploader(
            "Choose a PDF file", type="pdf", help="Upload your CV/Resume in PDF format"
        )

        pdf_text = ""
        if uploaded_file is not None:
            lines_with_coords = []
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

            fontsize = None

            for page_num, page in enumerate(doc, start=1):
                blocks = page.get_text("dict", sort=True)["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            x0, y0, x1, y1 = None, None, None, None
                            for span in line["spans"]:
                                line_text += span["text"]
                                # Update bbox for the line
                                sx0, sy0, sx1, sy1 = span["bbox"]
                                fontsize = span["size"]
                                if x0 is None:
                                    x0, y0, x1, y1 = sx0, sy0, sx1, sy1
                                else:
                                    x0 = min(x0, sx0)
                                    y0 = min(y0, sy0)
                                    x1 = max(x1, sx1)
                                    y1 = max(y1, sy1)
                            if line_text.strip():
                                lines_with_coords.append(
                                    {
                                        "text": line_text.strip(),
                                        "bbox": (x0, y0, x1, y1),
                                        "page": page_num,
                                    }
                                )
            for page in doc:
                print("here" * 20)
                paths = page.get_drawings()  # vector graphics on page
                bullets = []  # bullet point graphics
                for path in paths:
                    rect = path["rect"]  # rectangle containing the graphic
                    # filter out if width and height are both less than font size
                    if rect.width <= fontsize and rect.height <= fontsize:
                        bullets.append(path)

            for idx, bullet in enumerate(bullets):
                print(f"Bullet {idx + 1}: {bullet['type']} at {bullet['rect']}")

            merged_lines = merge_lines_by_bullets(lines_with_coords, bullets)

            doc.close()

            with fitz.open(stream=uploaded_file.getvalue()) as doc:
                page_number = st.sidebar.number_input(
                    "Page number",
                    min_value=1,
                    max_value=doc.page_count,
                    value=1,
                    step=1,
                )
                page = doc.load_page(page_number - 1)
                for line in merged_lines:
                    page.add_rect_annot(
                        line.get("bbox")
                    )  # Add annotation to the first block

                pix = page.get_pixmap(dpi=120).tobytes()
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
            disabled=not (pdf_text and user_text),
        )

    # Show requirements if button is disabled
    if not (pdf_text and user_text):
        st.warning(
            "⚠️ Please upload a PDF file AND add additional information to enable analysis."
        )

    # Perform analysis
    if analyze_button and pdf_text and user_text:
        st.subheader("🎯 AI Analysis Results")

        with st.spinner(f"Analyzing with {model_type}... This may take a moment."):
            analysis_result = get_llm_response(
                model_type, temperature, pdf_text, user_text
            )

        # Display results
        if analysis_result:
            st.markdown("### 📊 Analysis Report")
            st.markdown(analysis_result)

            # Download button for results
            st.download_button(
                label="📥 Download Analysis",
                data=analysis_result,
                file_name=f"cv_analysis_{model_type.lower()}.txt",
                mime="text/plain",
            )
        else:
            st.error(
                "❌ Failed to generate analysis. Please check your API configuration."
            )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit | Supports OpenAI, Claude, and Groq models</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
