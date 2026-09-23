import sqlite3
from datetime import datetime
from config.settings import DB_PATH

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def initialize():
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS assessments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            farmer_name TEXT,
            village TEXT,
            boundary_source TEXT,
            field_area_ha REAL,
            watershed_area_ha REAL,
            lulc_name TEXT,
            field_cn REAL,
            field_slope REAL,
            field_runoff_mm REAL,
            field_runoff_m3 REAL,
            stream_distance_m REAL,
            recommendations TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_assessment(farmer_name, village, result):
    conn = connect()
    conn.execute("""
        INSERT INTO assessments(
            created_at, farmer_name, village, boundary_source,
            field_area_ha, watershed_area_ha, lulc_name, field_cn,
            field_slope, field_runoff_mm, field_runoff_m3,
            stream_distance_m, recommendations
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(),
        farmer_name,
        village,
        result.get("field_boundary_source"),
        result.get("field_area_ha"),
        result.get("watershed_area_ha"),
        result.get("dominant_lulc_name"),
        result.get("field_cn"),
        result.get("field_slope"),
        result.get("field_runoff_mm"),
        result.get("field_runoff_m3"),
        result.get("field_stream_distance_m"),
        "; ".join(x[0] for x in result.get("farmer_recommendations", []))
    ))
    conn.commit()
    conn.close()
