import os
import io
import re
import cv2
import base64
import smtplib
import requests
import numpy as np
import pytz
from datetime import datetime
from email.message import EmailMessage
from PIL import Image as PILImage
from ultralytics import YOLO
from playwright.sync_api import sync_playwright

# ============================================================
# CONFIGURATION & SECRETS
# ============================================================
ZENPUT_API_KEY = os.environ.get('ZENPUT_API_KEY')
GMAIL_APP_PASS = os.environ.get('GMAIL_APP_PASSWORD')
SENDER_EMAIL   = "aof.group.auto@gmail.com"

MODEL_PATH     = "best.pt"  # Make sure best.pt is in repo root

TZ    = pytz.timezone("Asia/Baghdad")
TODAY =  (datetime.now(TZ) - timedelta(days=1)).strftime("%Y-%m-%d")

TEMPLATES = {
    "Shawarma Classic": 401648,
    "Lubda":            472189,
    "Garatis":          671643,
}

BRAND_AR = {
    "Shawarma Classic": "شاورما كلاسيك ",
    "Lubda":            "لبدة ",
    "Garatis":          "قراطيس ",
}

EXCLUDED_BRANCH_CODES = {"B22", "B28", "B33", "B30", "QB04", "QB05", "QB07"}

RECIPIENTS_TO = [
    "a.alsalem@aofgroup.com"
]
RECIPIENTS_CC = [
    "o.salahaddin@aofgroup.com"
]

# ============================================================
# HELPERS
# ============================================================
def is_excluded_branch(branch_raw: str) -> bool:
    if not branch_raw:
        return False
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", str(branch_raw).upper()) if t]
    return any(t in EXCLUDED_BRANCH_CODES for t in tokens)

def zenput_headers():
    return {"X-API-TOKEN": ZENPUT_API_KEY, "Accept": "application/json"}

def get_signed_url(s3_path):
    if not s3_path: return ""
    url = f"https://www.zenput.com/api/v2/users/current/storage/?path={requests.utils.quote(s3_path)}"
    try:
        resp = requests.get(url, headers=zenput_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json().get("data", {}).get("location", "")
    except Exception:
        pass
    return ""

def download_image(url):
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return PILImage.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    return None

def image_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=80)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def convert_html_to_pdf(html_path, pdf_path):
    """Converts HTML file to A4 PDF using Playwright Chromium."""
    print(f"📄 Converting {html_path} to PDF...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{os.path.abspath(html_path)}")
        page.pdf(path=pdf_path, format="A4", print_background=True, margin={"top":"20px","bottom":"20px","left":"20px","right":"20px"})
        browser.close()
    print(f"✅ PDF Created: {pdf_path}")

# ============================================================
# FETCH TODAY'S SUBMISSIONS
# ============================================================
def fetch_today_photos():
    records = []
    print(f"📅 Fetching TODAY's data ({TODAY})...")
    
    for brand, tid in TEMPLATES.items():
        params = {"form_template_id": tid, "limit": 100, "date_submitted_start": TODAY}
        resp = requests.get("https://www.zenput.com/api/v3/submissions/", headers=zenput_headers(), params=params)
        if resp.status_code != 200: continue
        
        data = resp.json().get("data", [])
        for sub in data:
            meta = sub.get("smetadata", {}) or {}
            date_local = meta.get("date_submitted_local", "")[:10]
            
            if not date_local.startswith(TODAY):
                continue
                
            location  = meta.get("location", {}).get("name", "Unknown Branch")
            
            # EXCLUDED BRANCH CHECK
            if is_excluded_branch(location):
                print(f"  ⏭️ Skipping excluded branch: {location}")
                continue

            submitter = meta.get("created_by", {}).get("display_name", "Unknown") if isinstance(meta.get("created_by"), dict) else "Unknown"
            
            for ans in sub.get("answers", []) or []:
                title = str(ans.get("title", "")).lower()
                if "open the gas box" in title or "take picture from inside" in title:
                    for val in ans.get("value", []) or []:
                        if isinstance(val, dict) and val.get("s3_key"):
                            s3_url = get_signed_url(val["s3_key"])
                            if s3_url:
                                records.append({"brand": brand, "branch": location, "submitter": submitter, "date": date_local, "url": s3_url})
                                
    print(f"✅ Total Photos Found for {TODAY}: {len(records)}")
    return records

# ============================================================
# MAIN AI PROCESS & EMAIL
# ============================================================
def main():
    print(f"🤖 Loading Object Detector from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    records = fetch_today_photos()
    if not records:
        print(f"⚠️ No submissions found for today ({TODAY}). Exiting.")
        return

    # Categorize by Brand
    brand_cards = {
        "Shawarma Classic": {"wrong": [], "right": []},
        "Lubda":            {"wrong": [], "right": []},
        "Garatis":          {"wrong": [], "right": []},
    }

    total_wrong = 0

    for idx, rec in enumerate(records, 1):
        print(f"[{idx}/{len(records)}] Processing {rec['branch']} ({rec['brand']})...")
        pil_img = download_image(rec["url"])
        if pil_img is None: continue
        
        # Conf=0.38 and IOU=0.40 eliminates false clamps and merges duplicates
        results = model(pil_img, conf=0.38, iou=0.40, verbose=False)[0]
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        open_count = 0
        closed_count = 0
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label_name = str(model.names[cls_id]).lower()
            
            # Swapped label mapping: 'valve_closed' in Roboflow = OPEN (Green)
            if "closed" in label_name:
                open_count += 1
                box_color = (0, 255, 0) # Green for Open
                label_text = "Open"
            else:
                closed_count += 1
                box_color = (0, 0, 255) # Red for Closed
                label_text = "Closed"
                
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(cv_img, label_text, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        drawn_pil = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        img_b64 = image_to_base64(drawn_pil)
        
        # QC Rules
        if closed_count > 0:
            badge_text = f"❌ WRONG ({closed_count} Closed Valve Detected)"
            badge_color = "#e74c3c"
            border_color = "#e74c3c"
            is_wrong = True
        elif open_count >= 3:
            badge_text = f"✅ RIGHT ({open_count} Open Valves)"
            badge_color = "#27ae60"
            border_color = "#27ae60"
            is_wrong = False
        else:
            badge_text = f"⚠️ INCOMPLETE ({open_count}/3 Valves Visible)"
            badge_color = "#e67e22"
            border_color = "#e67e22"
            is_wrong = True
            
        card = f"""
        <div style="border: 3px solid {border_color}; border-radius: 10px; padding: 12px; width: 330px; background: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); page-break-inside: avoid;">
            <div style="background: #0f172a; border-radius: 6px; height: 260px; display: flex; align-items: center; justify-content: center;">
                <img src="data:image/jpeg;base64,{img_b64}" style="max-width: 100%; max-height: 260px; object-fit: contain; border-radius: 6px;">
            </div>
            <div style="margin-top: 10px; font-family: Arial, sans-serif;">
                <div style="font-size: 15px; font-weight: bold; color: #2c3e50;">{rec['branch']}</div>
                <div style="font-size: 12px; color: #7f8c8d;">{rec['submitter']} | <b>{rec['date']}</b></div>
                <div style="margin-top: 8px; padding: 6px 12px; background: {badge_color}; color: white; font-weight: bold; font-size: 13px; text-align: center; border-radius: 5px;">
                    {badge_text}
                </div>
            </div>
        </div>
        """
        
        b = rec['brand']
        if b in brand_cards:
            if is_wrong:
                brand_cards[b]["wrong"].append(card)
                total_wrong += 1
            else:
                brand_cards[b]["right"].append(card)

    # BUILD SECTIONS BY BRAND (Wrong cards first inside each section)
    sections_html = []
    for brand_name, cards in brand_cards.items():
        all_brand_cards = cards["wrong"] + cards["right"]
        if not all_brand_cards:
            continue
            
        ar_title = BRAND_AR.get(brand_name, brand_name)
        sec = f"""
        <div style="margin-top: 30px; margin-bottom: 30px;">
            <h2 style="color: #1a3c5e; border-bottom: 2px solid #1a3c5e; padding-bottom: 8px;">{ar_title} ({len(all_brand_cards)} فرع)</h2>
            <div style="display: flex; flex-wrap: wrap; gap: 16px; direction: ltr;">
                {"".join(all_brand_cards)}
            </div>
        </div>
        """
        sections_html.append(sec)

    # Build Full HTML Document
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Gas Box AI Inspection Report - {TODAY}</title>
    </head>
    <body style="font-family: Arial, sans-serif; background-color: #f8f9fa; padding: 20px;" dir="rtl">
        <div style="background: #1a3c5e; color: white; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 22px;">📦 تقرير فحص صندوق الغاز الذكي (AI) — {TODAY}</h1>
            <p style="margin: 5px 0 0 0; font-size: 14px;">إجمالي الفروع المفحوصة: <b>{len(records)}</b> | المخالفة/غير المكتملة: <b style="color: #ff8a80;">{total_wrong}</b></p>
        </div>
        
        {"".join(sections_html)}
    </body>
    </html>
    """

    html_filename = f"GasBox_AI_Report_{TODAY}.html"
    pdf_filename  = f"GasBox_AI_Report_{TODAY}.pdf"

    # Save HTML File
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_report)

    # Convert HTML to PDF
    convert_html_to_pdf(html_filename, pdf_filename)

    # SEND EMAIL WITH BOTH ATTACHMENTS
    print("📧 Sending Email Report with HTML and PDF attachments...")
    msg = EmailMessage()
    msg["Subject"] = f"📦 تقرير فحص صندوق الغاز الذكي (AI) - {TODAY}"
    msg["From"]    = f"Business Intelligence <{SENDER_EMAIL}>"
    msg["To"]      = ", ".join(RECIPIENTS_TO)
    msg["Cc"]      = ", ".join(RECIPIENTS_CC)
    
    # HTML Body in email
    msg.add_alternative(html_report, subtype="html")

    # Attach HTML File
    with open(html_filename, "rb") as f:
        msg.add_attachment(f.read(), maintype="text", subtype="html", filename=html_filename)

    # Attach PDF File
    with open(pdf_filename, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=pdf_filename)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SENDER_EMAIL, GMAIL_APP_PASS)
        smtp.send_message(msg)

    print("🎉 Email Sent Successfully with both HTML and PDF!")

if __name__ == "__main__":
    main()
