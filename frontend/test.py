import streamlit as st

# Function to display the image on hover
def display_image_on_hover(image_url, i):
    # Generate unique class names for each image
    hover_class = f'hoverable_{i}'
    tooltip_class = f'tooltip_{i}'
    image_popup_class = f'image-popup_{i}'

    # Define the unique CSS for each image
    hover_css = f'''
        .{hover_class} {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}
        .{hover_class} .{tooltip_class} {{
            opacity: 0;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            transition: opacity 0.5s;
            background-color: rgba(0, 0, 0, 0.8);
            color: #fff;
            padding: 4px;
            border-radius: 4px;
            text-align: center;
            white-space: nowrap;
        }}
        .{hover_class}:hover .{tooltip_class} {{
            opacity: 1;
        }}
        .{image_popup_class} {{
            position: absolute;
            display: none;
            background-image: none;
            width: 500px;
            height: 500px;
        }}
        .{hover_class}:hover .{image_popup_class} {{
            display: block;
            background-image: url({image_url});
            background-size: cover;
            z-index: 999;
        }}
    '''
    tooltip_css = f"<style>{hover_css}</style>"

    # Define the html for each image
    image_hover = f'''
        <div class="{hover_class}">
            <a href="{image_url}">{image_url}</a>
            <div class="{tooltip_class}">Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum</div>
            <div class="{image_popup_class}"></div>
        </div>
    '''
    
    # Write the dynamic HTML and CSS to the content container
    st.markdown(f'<p>{image_hover}{tooltip_css}</p>', unsafe_allow_html=True)

# Initialize Streamlit app
st.title("Floating Image Tooltips on MouseOver")

# A list of image urls
url_list = [
    "https://everydayswag.org/images/misc/ableton-sampler-vol-env.png",
    "https://everydayswag.org/images/email.png"
]

# Create a container for the content that triggers the tooltip on mouseover
with st.expander("⛓", expanded=True):
    # Generate the dynamic HTML and CSS for each URL in the list
    for i, url in enumerate(url_list):
        display_image_on_hover(url, i)