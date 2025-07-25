import numpy as np
from pydantic import BaseModel
from typing import List, Tuple
from pymupdf import Rect, Document


class LineWithCoords(BaseModel):
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page: int
    fontsize: float | None = None  # Optional font size


def bbox_vertically_close(line_bbox, bullet_rect, tolerance=5):
    # Check if the vertical center of the bullet is close to the line's vertical span
    _, line_y0, _, line_y1 = line_bbox
    _, bullet_y0, _, bullet_y1 = bullet_rect
    return (bullet_y0 >= line_y0 - tolerance) and (bullet_y1 <= line_y1 + tolerance)


def merge_lines_by_bullets(
    lines_with_coords: List[LineWithCoords],
    bullets: List[Rect],
    tolerance: int = 5,
    line_gap_factor: float = 1.1,
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
        page = 1
        bullets_by_page.setdefault(page, []).append(bullet)

    for page, page_bullets in bullets_by_page.items():
        # Sort bullets by their top y coordinate
        page_bullets = sorted(page_bullets, key=lambda b: b.y0)
        # Get all lines on this page, sorted by their top y
        page_lines = [
            (idx, line)
            for idx, line in enumerate(lines_with_coords)
            if line.page == page
        ]
        page_lines = sorted(page_lines, key=lambda x: x[1].bbox[1])

        # Estimate normal line gap for this page
        line_gaps = []
        for i in range(1, len(page_lines)):
            prev_y1 = page_lines[i - 1][1].bbox[3]
            curr_y0 = page_lines[i][1].bbox[1]
            line_gaps.append(curr_y0 - prev_y1)

        normal_gap = (
            np.median([gap for gap in line_gaps if gap > 0]) if line_gaps else 0
        )

        for i, bullet in enumerate(page_bullets):
            bullet_y0 = bullet.y0
            # Determine the y0 of the next bullet, or end of page
            if i + 1 < len(page_bullets):
                next_bullet_y0 = page_bullets[i + 1].y0
            else:
                next_bullet_y0 = float("inf")
            group_lines = []
            group_bbox = None
            last_line_y1 = None
            bullet_fontsize = None  # <-- change here
            for idx, line in page_lines:
                if idx in used:
                    continue
                line_y0 = line.bbox[1]
                line_y1 = line.bbox[3]
                # Only include lines whose top y is equal or after the bullet y
                if (line_y0 >= bullet_y0 - tolerance) and (
                    line_y0 < next_bullet_y0 - tolerance
                ):
                    # If this is the first line in the group, set bullet_fontsize
                    if bullet_fontsize is None:
                        bullet_fontsize = line.fontsize
                    # If this is not the first line in the group, check the gap and font size
                    if last_line_y1 is not None and normal_gap > 0:
                        gap = line_y0 - last_line_y1
                        if gap > line_gap_factor * normal_gap:
                            # Large gap, stop merging for this bullet
                            break
                    if (
                        bullet_fontsize is not None
                        and "fontsize" in line
                        and line.fontsize != bullet_fontsize
                    ):
                        # Font size changed, stop merging for this bullet
                        break

                    group_lines.append(line)
                    used.add(idx)
                    x0, y0, x1, y1 = line.bbox
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
                merged_text = " ".join([l.text for l in group_lines])
                merged.append(
                    LineWithCoords(
                        text=merged_text,
                        bbox=tuple(group_bbox),
                        page=page,
                        fontsize=bullet_fontsize,
                    )
                )
    # Add lines not assigned to any bullet
    for idx, line in enumerate(lines_with_coords):
        if idx not in used:
            merged.append(line)

    return merged


def get_lines_with_coords(doc: Document) -> Tuple[List[LineWithCoords], int]:
    """Extract lines with coordinates and font size from a PyMuPDF document."""
    lines_with_coords = []
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
                            LineWithCoords(
                                text=line_text.strip(),
                                bbox=(x0, y0, x1, y1),
                                page=page_num,
                                fontsize=fontsize,
                            )
                        )
    return lines_with_coords, fontsize


# code from https://stackoverflow.com/questions/75745643/cant-take-bulletpoints-from-pdf-using-python-fitz
def get_bullets_from_doc(doc: Document, fontsize: int) -> List[Rect]:
    """Extract bullet points from a PyMuPDF document using drawing objects and font size."""
    bullets = []  # bullet point graphics
    page = doc[0]  # page with number 'pno'
    paths = page.get_drawings()  # vector graphics on page
    for path in paths:
        rect = path["rect"]  # rectangle containing the graphic
        # filter out if width and height are both less than font size
        if rect.width <= fontsize and rect.height <= fontsize:
            bullets.append(rect)

    return bullets
