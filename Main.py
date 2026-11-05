import numpy as np
import pandas as pd
from scipy.stats import weibull_min
# หมายเหตุ: หากใช้ library 'reliability' ของ Python สามารถ import มาใช้ร่วมด้วยได้ เช่น from reliability.Fitters import Fit_Weibull_2P

# ==========================================
# 1. DATA PREPARATION & RANKING MODULE
# ==========================================

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    โหลดข้อมูลประวัติการเสียและทำความสะอาดข้อมูลเบื้องต้น
    """
    df = pd.read_csv(file_path) # หรือ pd.read_excel ตามรูปแบบไฟล์
    
    # แปลงคอลัมน์วันที่ให้เป็น datetime
    df['breakdown_time'] = pd.to_datetime(df['breakdown_time'])
    df['repair_start'] = pd.to_datetime(df['repair_start'])
    df['repair_end'] = pd.to_datetime(df['repair_end'])
    
    # คำนวณ MTTR (นาที) และ Downtime
    df['repair_duration_min'] = (df['repair_end'] - df['repair_start']).dt.total_seconds() / 60
    
    return df

def rank_parts_by_criticality(df: pd.DataFrame, top_n: int = 5) -> list:
    """
    จัดอันดับชิ้นส่วนตามความถี่ในการเสียและเวลาหยุดรวม เพื่อเลือกชิ้นส่วนนำร่อง 3-5 รายการ[cite: 1]
    """
    criticality_summary = df.groupby(['part_id', 'part_name']).agg(
        failure_count=('breakdown_time', 'count'),
        total_downtime_min=('repair_duration_min', 'sum'),
        avg_mttr=('repair_duration_min', 'mean')
    ).reset_index()
    
    # จัดอันดับเรียงตามความถี่และเวลาหยุดรวมสูงสุด
    ranked_parts = criticality_summary.sort_values(
        by=['failure_count', 'total_downtime_min'], 
        ascending=False
    )
    
    # คืนค่ารายการชิ้นส่วน Top N นำร่อง
    pilot_parts = ranked_parts.head(top_n)['part_id'].tolist()
    return pilot_parts


# ==========================================
# 2. RELIABILITY & WEIBULL ANALYSIS MODULE
# ==========================================

def calculate_time_between_failures(df: pd.DataFrame, part_id: str) -> np.ndarray:
    """
    คำนวณเวลาการทำงานระหว่างการเสีย (Time Between Failures - TBF) ของชิ้นส่วนที่กำหนด
    """
    part_df = df[df['part_id'] == part_id].sort_values('breakdown_time')
    # คำนวณระยะเวลาระหว่างการเสียแต่ละครั้ง (ชั่วโมง)
    tbf = part_df['breakdown_time'].diff().dt.total_seconds().dropna() / 3600
    return tbf.values

def fit_weibull_distribution(tbf_data: np.ndarray):
    """
    วิเคราะห์ Bathtub Curve ด้วย Weibull Distribution (หาค่า Shape parameter - Beta)
    Beta < 1: ช่วงต้น (Infant Mortality)
    Beta = 1: เสียแบบสุ่ม (Random Failures)
    Beta > 1: เสื่อมสภาพตามอายุ (Wear-out)
    """
    if len(tbf_data) < 3:
        return None, "ข้อมูลน้อยเกินไป ไม่สามารถfit Weibull ได้"
    
    # Fit ข้อมูลด้วย Weibull Distribution (scipy หรือ reliability library)
    c, loc, scale = weibull_min.fit(tbf_data, floc=0)
    beta = c  # Shape parameter
    eta = scale # Scale parameter (Characteristic life)
    
    return {"beta": beta, "eta": eta}, get_failure_mode_description(beta)

def get_failure_mode_description(beta: float) -> str:
    """
    แปลผลค่า Beta เป็นรูปแบบการเสียตาม Bathtub Curve[cite: 1]
    """
    if beta < 1.0:
        return "Early-life failure (ช่วงต้น - เสียเพราะความบกพร่องในการติดตั้งหรือการผลิต)"
    elif 0.9 <= beta <= 1.1:
        return "Random failure (เสียแบบสุ่ม - เกิดจากปัจจัยภายนอก)"
    else:
        return "Wear-out failure (เสื่อมสภาพตามอายุการใช้งาน)"


# ==========================================
# 3. PM CATEGORIZATION MODULE
# ==========================================

def categorize_pm_strategy(beta: float, part_characteristics: str) -> str:
    """
    จัดประเภท PM ตามรูปแบบการเสียและลักษณะของชิ้นส่วน[cite: 1]
    """
    if beta > 1.2:
        return "Time-based / Value-based PM (ชิ้นส่วนเสื่อมสภาพ กำหนดรอบเปลี่ยนชัดเจนได้)"
    else:
        return "Condition-based PM / Safety Stock (ชิ้นส่วนเสียแบบสุ่ม ต้องอาศัยการตรวจวัดหรือเตรียมสต็อกปลอดภัย)"


# ==========================================
# 4. INVENTORY OPTIMIZATION MODULE
# ==========================================

def calculate_min_max_inventory(avg_daily_demand: float, lead_time_days: float, safety_stock: float) -> dict:
    """
    คำนวณระดับอะไหล่ต่ำสุดและสูงสุด (Min-Max Inventory Levels)[cite: 1]
    """
    reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
    max_stock = reorder_point + (avg_daily_demand * lead_time_days) # ตัวอย่างสูตรพื้นฐาน ปรับเปลี่ยนได้ตามนโยบายบริษัท
    
    return {
        "reorder_point": round(reorder_point, 2),
        "max_stock": round(max_stock, 2),
        "safety_stock": round(safety_stock, 2)
    }


# ==========================================
# 5. KPI TRACKING MODULE
# ==========================================

def calculate_kpis(df: pd.DataFrame, pilot_parts: list) -> dict:
    """
    คำนวณค่าตัวชี้วัด MTTR และภาพรวมของชิ้นส่วนนำร่อง[cite: 1]
    """
    filtered_df = df[df['part_id'].isin(pilot_parts)]
    
    avg_mttr = filtered_df['repair_duration_min'].mean()
    total_stockout_events = filtered_df['waiting_for_parts'].sum() if 'waiting_for_parts' in filtered_df.columns else 0
    
    return {
        "average_mttr_minutes": round(avg_mttr, 2),
        "stockout_incidents": int(total_stockout_events)
    }