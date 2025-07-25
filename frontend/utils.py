import fitz

def duplicate_fitz_page(page):
    """
    Create a duplicate of a PyMuPDF page object.
    Returns a new page object with the same content.
    """
    # Export the page as a PDF bytes
    pdf_bytes = page.parent.write()
    # Open a new document from the single-page PDF
    new_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Return the first (and only) page of the new document
    return new_doc.load_page(0)
