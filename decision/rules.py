import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = json.loads(
    (ROOT / "config" / "structure_rules.json").read_text(encoding="utf-8")
)

def recommend_field(area_ha, slope, runoff_mm, cn, stream_distance, lulc_name):
    lname = (lulc_name or "").lower()
    out = []

    if "built" in lname or "settlement" in lname:
        return [
            ("Rooftop Rainwater Harvesting",
             "Built-up land is better suited to rooftop collection than agricultural excavation."),
            ("Recharge Pit",
             "Recharge can be screened where soil and groundwater conditions are suitable.")
        ]

    if "waterbody" in lname or "water body" in lname:
        return [
            ("Existing Water-body Improvement",
             "Screen renovation, desiltation and inlet/outlet improvement rather than a duplicate pond.")
        ]

    if "river" in lname or "stream" in lname:
        return [
            ("Drainage-line Assessment",
             "The selected land-use class is a stream/river. Use watershed-level structure screening.")
        ]

    if stream_distance is not None and stream_distance <= RULES["offstream_diversion_max_m"]:
        out.append((
            "Off-stream Farm Pond with Controlled Diversion",
            "Field is very close to a mapped stream; diversion requires hydraulic verification."
        ))
    elif stream_distance is not None and stream_distance <= RULES["offstream_pond_max_m"]:
        out.append((
            "Off-stream Farm Pond",
            "Mapped stream is within 200 m; off-stream storage can be screened."
        ))

    agricultural = any(
        k in lname
        for k in ["crop", "plantation", "shifting cultivation"]
    )

    fp = RULES["farm_pond"]
    if agricultural and np.isfinite(slope) and np.isfinite(runoff_mm):
        if slope <= fp["max_mean_slope_pct"] and runoff_mm >= fp["min_runoff_mm"]:
            out.append((
                "Farm Pond",
                "Agricultural LULC, gentle slope and available runoff support farm-pond screening."
            ))

    rp = RULES["recharge_pond"]
    if np.isfinite(slope) and np.isfinite(cn):
        if slope <= rp["max_mean_slope_pct"] and cn <= rp["max_cn"]:
            out.append((
                "Recharge / Percolation Pond",
                "Gentle terrain and comparatively lower CN support recharge-oriented screening."
            ))

    ct = RULES["contour_trench"]
    if np.isfinite(slope) and ct["min_slope_pct"] < slope <= ct["max_slope_pct"]:
        out.append((
            "Contour Trenches / Field Bunds",
            "Moderate slope favours distributed contour runoff interception."
        ))

    if np.isfinite(slope) and slope > RULES["staggered_trench"]["min_slope_pct"]:
        out.append((
            "Staggered Trenches / Vegetative Barriers",
            "Steeper terrain favours distributed soil and water conservation measures."
        ))

    sp = RULES["small_collection_pit"]
    if area_ha <= sp["max_field_area_ha"] and np.isfinite(runoff_mm):
        if runoff_mm >= sp["min_runoff_mm"]:
            out.append((
                "Small Collection / Recharge Pit",
                "Small field area and usable runoff support distributed storage/recharge."
            ))

    if "forest" in lname:
        out = [
            x for x in out
            if x[0] not in {"Farm Pond", "Off-stream Farm Pond"}
        ]
        out.append((
            "Vegetative / Recharge-zone Treatment",
            "Forest land should prioritise low-disturbance recharge and vegetative treatment."
        ))

    if "scrub" in lname:
        out.append((
            "Staggered Trench / Vegetative Treatment",
            "Scrub land is suited to distributed runoff reduction and revegetation screening."
        ))

    if not out:
        out.append((
            "In-situ Rainwater Conservation",
            "Current LULC and hydrological indicators favour distributed conservation."
        ))

    unique = []
    seen = set()

    for item in out:
        if item[0] not in seen:
            unique.append(item)
            seen.add(item[0])

    return unique

def recommend_watershed(lulc_comp):
    out = [
        (
            "Check Dam / Small Drainage-line Structure",
            "A mapped drainage and DEM-derived watershed are available. "
            "Final location requires channel cross-section, foundation and spillway design."
        )
    ]

    if lulc_comp:
        ag = sum(
            x["percent"]
            for x in lulc_comp
            if any(k in x["name"].lower() for k in [
                "crop", "plantation", "shifting cultivation"
            ])
        )

        forest = sum(
            x["percent"]
            for x in lulc_comp
            if "forest" in x["name"].lower()
        )

        if ag > 30:
            out.append((
                "Agricultural Contour Treatment",
                "Substantial agricultural area supports bunds, contour measures and distributed farm storage."
            ))

        if forest > 25:
            out.append((
                "Upper-watershed Vegetative Treatment",
                "Forest-dominated portions favour low-disturbance recharge and vegetative conservation."
            ))

    return out
