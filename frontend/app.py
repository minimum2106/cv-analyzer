import streamlit as st
import pymupdf 
import io
from typing import Optional, Dict, Any
import openai
import anthropic
from groq import Groq

# Configure page
st.set_page_config(
    page_title="CV Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
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

def get_llm_response(model_type: str, temperature: float, pdf_text: str, user_text: str) -> str:
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
                max_tokens=1500
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
                messages=[{"role": "user", "content": prompt}]
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
                max_tokens=1500
            )
            return response.choices[0].message.content
            
    except Exception as e:
        return f"Error getting response from {model_type}: {str(e)}"

def main():
    # Initialize session state
    if 'sidebar_collapsed' not in st.session_state:
        st.session_state.sidebar_collapsed = False
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 AI Settings")
        
        # Model selection
        st.subheader("Select LLM Model")
        model_type = st.selectbox(
            "Choose your AI model:",
            ["OpenAI", "Claude", "Groq"],
            help="Select which AI model to use for analysis"
        )
        
        # Temperature slider
        st.subheader("Model Parameters")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness: 0.0 = focused, 2.0 = creative"
        )
        
        # Model info
        st.info(f"**Selected Model:** {model_type}\n**Temperature:** {temperature}")
        
        # Sidebar collapse button
        if st.button("🔄 Toggle Sidebar"):
            st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
            st.rerun()
    
    # Main content
    st.title("📄 CV/Resume Analyzer")
    st.markdown("Upload your CV/Resume and add additional information for AI-powered analysis.")
    
    # Create two columns for the main content
    col1, col2 = st.columns(2)
    
    # Left column - PDF Upload
    with col1:
        st.subheader("📁 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload your CV/Resume in PDF format"
        )
        
        pdf_text = ""
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"File size: {uploaded_file.size} bytes")
            
            # Extract text from PDF
            with st.spinner("Extracting text from PDF..."):
                pdf_text = extract_pdf_text(uploaded_file)
            
            if pdf_text:
                st.success("✅ Text extracted successfully!")
                
                # Show preview of extracted text
                with st.expander("📖 Preview Extracted Text"):
                    st.text_area(
                        "Extracted content:",
                        pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text,
                        height=200,
                        disabled=True
                    )
            else:
                st.error("❌ Failed to extract text from PDF")
    
    # Right column - Text Input
    with col2:
        st.subheader("✏️ Additional Information")
        user_text = st.text_area(
            "Add any additional information:",
            placeholder="Enter job descriptions, specific requirements, career goals, or any other relevant information...",
            height=300,
            help="Provide context or specific questions for the AI analysis"
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
            disabled=not (pdf_text and user_text)
        )
    
    # Show requirements if button is disabled
    if not (pdf_text and user_text):
        st.warning("⚠️ Please upload a PDF file AND add additional information to enable analysis.")
    
    # Perform analysis
    if analyze_button and pdf_text and user_text:
        st.subheader("🎯 AI Analysis Results")
        
        with st.spinner(f"Analyzing with {model_type}... This may take a moment."):
            analysis_result = get_llm_response(model_type, temperature, pdf_text, user_text)
        
        # Display results
        if analysis_result:
            st.markdown("### 📊 Analysis Report")
            st.markdown(analysis_result)
            
            # Download button for results
            st.download_button(
                label="📥 Download Analysis",
                data=analysis_result,
                file_name=f"cv_analysis_{model_type.lower()}.txt",
                mime="text/plain"
            )
        else:
            st.error("❌ Failed to generate analysis. Please check your API configuration.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit | Supports OpenAI, Claude, and Groq models</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
