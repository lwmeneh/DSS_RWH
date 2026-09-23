import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph,
    Spacer, Table, TableStyle
)
from reportlab.lib import colors

def make_pdf(farmer_name, village, r):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("Ri Bhoi Rainwater Harvesting GIS DSS", styles["Title"]),
        Paragraph("Automatic Smart Assessment Report", styles["Heading2"]),
        Spacer(1, 8)
    ]

    rows = [
        ["Farmer", farmer_name],
        ["Village", village],
        ["Boundary source", r.get("field_boundary_source", "")],
        ["Dominant LULC", r.get("dominant_lulc_name", "")],
        ["Field area", f'{r.get("field_area_ha",0):.3f} ha'],
        ["Mean CN-II", f'{r.get("field_cn",float("nan")):.1f}'],
        ["Mean slope", f'{r.get("field_slope",float("nan")):.2f}%'],
        ["Annual runoff", f'{r.get("field_runoff_mm",float("nan")):.1f} mm'],
        ["Annual runoff volume", f'{r.get("field_runoff_m3",float("nan")):,.0f} m³'],
    ]

    if r.get("watershed_area_ha") is not None:
        rows.append([
            "Watershed area",
            f'{r["watershed_area_ha"]:.2f} ha'
        ])

    table = Table(rows, colWidths=[180, 300])
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (0,-1), colors.whitesmoke),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("PADDING", (0,0), (-1,-1), 6)
    ]))

    story += [
        table,
        Spacer(1, 10),
        Paragraph("Farmer-level recommendations", styles["Heading2"])
    ]

    for name, reason in r.get("farmer_recommendations", []):
        story += [
            Paragraph(f"<b>{name}</b>: {reason}", styles["BodyText"]),
            Spacer(1, 4)
        ]

    if r.get("watershed_recommendations"):
        story += [
            Spacer(1, 8),
            Paragraph("Watershed-level recommendations", styles["Heading2"])
        ]
        for name, reason in r["watershed_recommendations"]:
            story += [
                Paragraph(f"<b>{name}</b>: {reason}", styles["BodyText"]),
                Spacer(1, 4)
            ]

    story += [
        Spacer(1, 10),
        Paragraph(
            "<b>Important:</b> automatic LULC detection identifies a mapped agricultural "
            "patch and not a cadastral/legal landholding boundary. Final structure location "
            "and design require field survey and engineering verification.",
            styles["BodyText"]
        )
    ]

    doc.build(story)
    buf.seek(0)
    return buf.read()
