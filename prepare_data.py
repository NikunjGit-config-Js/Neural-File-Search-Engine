"""Create the demonstration corpus, queries and relevance labels.

The documents describe a FICTIONAL parcel-delivery company ("Brightfield
Logistics"). They were written for this project so the search engine can be
tested on realistic internal documents in all four supported formats.

Queries and relevance labels are defined here, by hand, together with the
documents and *before* running any retrieval model, so the labels do not
depend on what either model returns.

Run:  python prepare_data.py
"""

import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from docx import Document
from pptx import Presentation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR / "sample_documents"
SYNTHETIC_NOTE = "Synthetic demonstration document created for the Neural File Search project."
FIXED_DATE = datetime(2026, 10, 1)  # fixed timestamps make the generated files byte-identical on every run


# ---------------------------------------------------------------------------
# File writers (also reused by the tests to build small fixture files)
# ---------------------------------------------------------------------------

def write_pdf(path, title, pages):
    """pages: list of (heading, paragraph). Each item is placed on its own page."""
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, title=title, author="Brightfield Logistics (fictional)",
                            subject=SYNTHETIC_NOTE, invariant=True)
    story = []
    for i, (heading, text) in enumerate(pages):
        if i == 0:
            story += [Paragraph(escape(title), styles["Title"]), Spacer(1, 12)]
        story += [Paragraph(escape(heading), styles["Heading2"]), Paragraph(escape(text), styles["BodyText"])]
        if i < len(pages) - 1:
            story.append(PageBreak())
    doc.build(story)


def write_docx(path, title, sections):
    """sections: list of (heading, paragraph)."""
    doc = Document()
    doc.core_properties.title = title
    doc.core_properties.comments = SYNTHETIC_NOTE
    doc.core_properties.created = doc.core_properties.modified = FIXED_DATE
    doc.add_heading(title, level=0)
    for heading, text in sections:
        doc.add_heading(heading, level=1)
        doc.add_paragraph(text)
    doc.save(path)
    fix_zip_timestamps(path)


def write_pptx(path, slides):
    """slides: list of (title, [bullet, ...])."""
    prs = Presentation()
    prs.core_properties.subject = SYNTHETIC_NOTE
    prs.core_properties.created = prs.core_properties.modified = FIXED_DATE
    for title, bullets in slides:
        if bullets:
            slide = prs.slides.add_slide(prs.slide_layouts[1])  # "Title and Content"
            slide.shapes.title.text = title
            body = slide.placeholders[1].text_frame
            body.text = bullets[0]
            for bullet in bullets[1:]:
                body.add_paragraph().text = bullet
        else:
            slide = prs.slides.add_slide(prs.slide_layouts[5])  # "Title Only"
            slide.shapes.title.text = title
    prs.save(path)
    fix_zip_timestamps(path)


def fix_zip_timestamps(path):
    """DOCX/PPTX files are zip archives that store the save time of every entry.
    Rewrite them with a fixed time so regenerating the corpus gives identical bytes."""
    with zipfile.ZipFile(path) as source:
        entries = [(info.filename, source.read(info)) for info in source.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=FIXED_DATE.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(info, data)


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# PDF documents
# ---------------------------------------------------------------------------

PDFS = {
    "employee_leave_policy.pdf": ("Employee Leave Policy", [
        ("Annual Leave",
         "All full-time employees accrue 25 days of paid annual leave per calendar year, earned at a rate of "
         "2.08 days per month. Part-time staff accrue leave on a pro-rata basis according to their contracted "
         "hours. Leave requests must be submitted through the HR portal at least two weeks in advance and "
         "approved by the line manager. Up to five unused days may be carried over into the first quarter of the "
         "following year; any remaining balance is forfeited on 31 March. During the peak season from "
         "mid-November to early January, leave for operations staff may be limited to protect delivery capacity."),
        ("Sick Leave",
         "Employees who are unwell must notify their manager before the start of their shift. Absences of up to "
         "three consecutive days can be self-certified. For absences longer than three days a medical certificate "
         "from a doctor is required. Employees receive full pay for the first ten days of sick leave in a rolling "
         "twelve-month period and statutory sick pay thereafter. Repeated short absences may lead to a "
         "confidential return-to-work conversation focused on support and wellbeing."),
        ("Parental and Family Leave",
         "Birth mothers are entitled to 26 weeks of maternity leave, of which the first 16 weeks are paid at full "
         "salary. Partners, including adoptive parents, receive four weeks of paternity leave at full pay, to be "
         "taken within six months of the birth or placement. Employees may also request up to five days of unpaid "
         "compassionate leave per year to care for a dependant or attend a funeral. Requests for flexible working "
         "on return from parental leave will be considered sympathetically."),
    ]),
    "information_security_policy.pdf": ("Information Security Policy", [
        ("Purpose and Scope",
         "This policy sets the minimum security requirements for every employee, contractor and temporary worker "
         "who uses Brightfield Logistics systems. It covers laptops, mobile phones, the warehouse management "
         "system, the customer portal and all cloud services. Breaches of this policy may result in disciplinary "
         "action."),
        ("Passwords and Authentication",
         "Passwords must be at least 14 characters long and must not be reused across systems. A password manager "
         "approved by IT is provided to all staff. Multi-factor authentication (MFA) is mandatory for email, VPN "
         "and any system that stores customer data. Accounts are locked after five failed sign-in attempts. "
         "Shared accounts are prohibited except for documented warehouse scanner terminals."),
        ("Phishing and Incident Reporting",
         "Staff must not open attachments or click links in unexpected messages that ask for credentials, payment "
         "or urgent action. Suspected phishing emails should be reported with the Report Phishing button in "
         "Outlook or forwarded to the security team. If you believe your account has been compromised, change "
         "your password immediately and call the IT service desk. All security incidents must be reported within "
         "one hour of discovery."),
        ("Data Classification",
         "Information is classified as Public, Internal, Confidential or Restricted. Customer addresses and "
         "payment details are Restricted and may only be stored in approved systems. Restricted data must be "
         "encrypted at rest and in transit and must never be copied to personal devices or USB drives."),
    ]),
    "travel_expense_policy.pdf": ("Travel and Expense Policy", [
        ("Booking Business Travel",
         "All business travel must be booked through the corporate travel portal. Economy class is standard for "
         "flights under six hours; premium economy may be approved for longer journeys. Rail travel is preferred "
         "for trips under 300 kilometres. Hotel bookings should not exceed the nightly cap of 150 EUR in regional "
         "cities and 220 EUR in capital cities."),
        ("Reimbursement of Expenses",
         "Out-of-pocket expenses are reimbursed through the expense management system within 30 days of the trip. "
         "Every claim must include an itemised receipt; card statements alone are not accepted. Claims submitted "
         "more than 60 days after the expense was incurred will not be paid. Mileage for private cars used on "
         "company business is reimbursed at 0.30 EUR per kilometre."),
        ("Meals and Per Diem",
         "A daily meal allowance (per diem) of 45 EUR applies to overnight trips within the country and 60 EUR "
         "for international travel. Alcohol is not reimbursable. Client entertainment must be pre-approved by a "
         "director and the names of attendees recorded on the claim."),
    ]),
    "q3_2026_financial_report.pdf": ("Third Quarter 2026 Financial Report", [
        ("Summary",
         "Brightfield Logistics reported third quarter 2026 revenue of 48.2 million EUR, an increase of 11 percent "
         "compared with the same quarter last year. Growth was driven by the new same-day delivery service in "
         "metropolitan areas and two large retail contracts signed in the spring. Operating profit rose to 4.1 "
         "million EUR, giving an operating margin of 8.5 percent."),
        ("Costs and Margins",
         "Fuel costs fell 6 percent year on year as the electric van pilot expanded and diesel prices stabilised. "
         "Staff costs increased 9 percent because of seasonal hiring and the new collective wage agreement. "
         "Warehouse rent rose following the opening of the Rotterdam cross-dock facility. Gross margin improved by "
         "1.2 percentage points."),
        ("Cash Flow and Outlook",
         "Free cash flow for the quarter was 2.7 million EUR. Capital expenditure of 3.9 million EUR was invested "
         "mainly in sorting equipment and charging infrastructure. Net debt decreased to 18 million EUR. For the "
         "full year management expects revenue between 185 and 190 million EUR, assuming peak season volumes in "
         "line with forecasts."),
    ]),
    "warehouse_safety_manual.pdf": ("Warehouse Safety Manual", [
        ("Personal Protective Equipment",
         "Safety boots with steel toe caps and high-visibility vests must be worn at all times on the warehouse "
         "floor. Cut-resistant gloves are required when handling strapping, pallets and sharp packaging. Hearing "
         "protection must be worn in the sorting hall when the conveyor system is running. Visitors are issued "
         "PPE at reception and must be escorted."),
        ("Forklifts and Material Handling",
         "Only staff holding a valid forklift certification may operate forklifts, reach trucks or pallet "
         "stackers. Certification is renewed every three years after a practical assessment. Pedestrians must use "
         "marked walkways and never walk under a raised load. Maximum speed inside the building is 8 km/h. Lift "
         "heavy boxes using your legs, not your back, and request assistance for anything over 25 kg."),
        ("Accidents and Emergencies",
         "All accidents, near misses and injuries must be recorded in the incident log before the end of the "
         "shift, however minor. First aid kits and defibrillators are located beside each loading dock. In case "
         "of fire, stop all equipment, leave by the nearest emergency exit and gather at the assembly point. "
         "Fire drills take place twice a year."),
    ]),
    "remote_work_guidelines.pdf": ("Remote Work Guidelines", [
        ("Eligibility",
         "Office-based employees may work remotely for up to three days per week with the agreement of their "
         "manager. Roles that require physical presence, such as warehouse operations, drivers and facilities "
         "staff, are not eligible. New starters are expected to work from the office for their first month to "
         "build relationships with their team."),
        ("Home Office Setup",
         "Remote workers receive a one-off equipment allowance of 400 EUR for a desk, chair or monitor, claimable "
         "through the expense system. The company laptop must be used for all work; personal computers must not "
         "be used to access company systems. A stable internet connection of at least 20 Mbps is required. "
         "Employees are responsible for keeping their workspace safe and ergonomic."),
        ("Working Hours and Communication",
         "Core collaboration hours are 10:00 to 15:00, during which remote staff must be reachable on Teams. "
         "Cameras should be switched on for team meetings where possible. Working from another country for more "
         "than ten days per year requires approval from HR because of tax and legal implications."),
    ]),
    "vendor_contract_summary.pdf": ("Contract Summary: Swiftline IT Services", [
        ("Overview",
         "Brightfield Logistics has contracted Swiftline IT Services to provide managed hosting and 24/7 support "
         "for the warehouse management system and the customer tracking portal. The agreement runs for three "
         "years from 1 February 2025 with an annual fee of 620,000 EUR, invoiced quarterly in advance."),
        ("Service Levels",
         "Swiftline guarantees 99.9 percent monthly availability for production systems. Priority 1 incidents "
         "must receive a response within 15 minutes and a resolution or workaround within four hours. If "
         "availability falls below the guaranteed level, Brightfield is entitled to service credits of 5 percent "
         "of the monthly fee for each 0.1 percent shortfall, capped at 30 percent."),
        ("Termination and Liability",
         "Either party may terminate for material breach with 60 days written notice if the breach is not "
         "remedied. Brightfield may terminate for convenience after the first 18 months with six months notice. "
         "Swiftline's total liability is capped at 150 percent of the annual fee, excluding liability for data "
         "protection breaches, which is unlimited. On exit, Swiftline must return all data and provide transition "
         "assistance for up to 90 days."),
    ]),
    "data_retention_policy.pdf": ("Data Retention Policy", [
        ("Principles",
         "We keep personal data only for as long as necessary for the purpose for which it was collected, in "
         "line with the General Data Protection Regulation (GDPR). Each data owner is responsible for applying "
         "the retention schedule to the systems they manage and for reviewing it annually."),
        ("Retention Schedule",
         "Customer delivery records, including names, addresses and proof-of-delivery signatures, are retained "
         "for 24 months after the delivery date. Invoices and financial records are retained for seven years to "
         "meet tax obligations. CCTV footage from warehouses is kept for 30 days unless required for an "
         "investigation. Recruitment files for unsuccessful candidates are deleted after six months."),
        ("Deletion and Subject Requests",
         "When the retention period ends, data must be securely erased or anonymised. Paper records are shredded "
         "using the confidential waste service. Requests from individuals to access or erase their personal data "
         "must be forwarded to the Data Protection Officer within two working days and answered within one month."),
    ]),
}

# ---------------------------------------------------------------------------
# DOCX documents
# ---------------------------------------------------------------------------

DOCXS = {
    "new_hire_onboarding_checklist.docx": ("New Hire Onboarding Checklist", [
        ("Before the First Day",
         "The hiring manager confirms the start date with HR and orders a laptop, phone and access badge at least "
         "ten working days before the new employee arrives. IT creates the user account and email address and "
         "assigns the required software licences. A welcome email with the address, start time and dress code is "
         "sent to the new starter."),
        ("First Day",
         "The new starter is met at reception by their buddy, collects their badge and completes the health and "
         "safety induction. IT hands over the laptop, helps set up multi-factor authentication and walks through "
         "the password manager. The manager introduces the team and explains the goals for the first 90 days."),
        ("First Week",
         "Complete the mandatory e-learning modules on information security, data protection and anti-bribery. "
         "Shadow colleagues in operations for at least half a day to understand how parcels move through the "
         "network. Book one-to-one meetings with key stakeholders and read the employee handbook."),
    ]),
    "incident_postmortem_tracking_outage.docx": ("Incident Postmortem: Tracking Portal Outage", [
        ("Summary",
         "On 14 August 2026 the customer parcel tracking portal was unavailable for 3 hours and 20 minutes between "
         "07:40 and 11:00 CET. Customers could not see delivery status, and the support team received over 2,300 "
         "calls and chats during the outage."),
        ("Root Cause",
         "A scheduled database migration added an index to the shipments table without using the online option. "
         "The table was locked, application connections piled up and the connection pool was exhausted. "
         "Automated health checks restarted the web servers repeatedly, which made the problem worse."),
        ("Resolution and Actions",
         "The on-call engineer cancelled the migration and restarted the database at 10:35, and service was fully "
         "restored by 11:00. Follow-up actions: require online schema changes for large tables, run migrations "
         "outside business hours, add alerts on connection pool usage, and update the status page within 15 "
         "minutes of any major incident."),
    ]),
    "marketing_campaign_plan.docx": ("Marketing Campaign Plan: Same-Day Delivery Launch", [
        ("Campaign Objective",
         "Launch the new same-day delivery service to small and medium online retailers in Amsterdam, Rotterdam "
         "and Utrecht, with a target of 400 new business accounts by the end of December 2026."),
        ("Channels and Messaging",
         "The campaign will combine LinkedIn advertising, search engine marketing, email newsletters to existing "
         "customers and a partnership with two e-commerce platform providers. The core message is that orders "
         "placed before noon arrive the same evening, with live tracking and a guaranteed two-hour delivery window."),
        ("Budget and Measurement",
         "The total budget is 180,000 EUR, with 45 percent allocated to paid social, 30 percent to search and 25 "
         "percent to events and content. Success will be measured by cost per acquired account, conversion rate "
         "from trial to paid contract and customer satisfaction after the first month."),
    ]),
    "customer_support_playbook.docx": ("Customer Support Playbook", [
        ("Tone and Response Times",
         "Agents should respond in a calm, friendly and professional tone, using the customer's name and avoiding "
         "internal jargon. Live chats must be answered within two minutes and emails within four business hours."),
        ("Handling Complaints and Refunds",
         "Listen to the full complaint before offering a solution and acknowledge the inconvenience. Agents can "
         "issue refunds of shipping fees up to 50 EUR without approval. Compensation above 50 EUR or claims for "
         "damaged goods must be escalated to a team lead. Always record the reason for the refund in the ticket so "
         "that recurring problems can be analysed."),
        ("Escalation Path",
         "Escalate to a team lead when a customer threatens legal action, when a parcel worth more than 1,000 EUR "
         "is lost, or when the same issue has been raised three times. Team leads escalate to the operations "
         "manager for depot-level failures. Never promise a delivery time that has not been confirmed by the depot."),
    ]),
    "performance_review_process.docx": ("Performance Review Process", [
        ("Annual Cycle",
         "Performance reviews take place once a year in January, with a lighter mid-year check-in in July. Each "
         "employee agrees three to five measurable objectives with their manager at the start of the year and "
         "records them in the HR system."),
        ("Ratings and Calibration",
         "Managers rate performance on a five-point scale from 'below expectations' to 'exceptional'. Ratings are "
         "discussed in calibration sessions with other managers in the same department to make sure standards are "
         "applied consistently and fairly."),
        ("Pay and Promotion",
         "Merit pay increases are linked to the final rating and the department budget, and take effect from 1 "
         "April. Promotion recommendations are reviewed by the leadership team twice a year. Employees who "
         "disagree with their rating can request a review meeting with their manager's manager and an HR partner."),
    ]),
    "procurement_procedure.docx": ("Procurement Procedure", [
        ("When This Procedure Applies",
         "This procedure applies to every purchase of goods or services made on behalf of Brightfield Logistics, "
         "including software subscriptions, vehicles, packaging materials and consultancy."),
        ("Approval Thresholds",
         "Purchases under 1,000 EUR can be approved by the budget holder. Purchases between 1,000 and 25,000 EUR "
         "require a purchase order approved by the department head and at least two written quotes. Purchases "
         "above 25,000 EUR require a competitive tender run by the procurement team and approval from the finance "
         "director."),
        ("Supplier Onboarding",
         "New suppliers must complete a due diligence questionnaire covering financial stability, insurance, data "
         "protection and sustainability practices. Bank details are verified by phone before the first payment to "
         "prevent invoice fraud. Preferred suppliers are reviewed annually against price, quality and delivery "
         "performance."),
    ]),
    "meeting_minutes_fleet_electrification.docx": ("Meeting Minutes: Fleet Steering Group", [
        ("Attendees and Purpose",
         "Fleet Steering Group, 9 September 2026. Present: Head of Fleet, Operations Director, Sustainability "
         "Lead, Finance Business Partner. Purpose: review the results of the electric van pilot in Utrecht and "
         "decide whether to extend it."),
        ("Pilot Results",
         "Twenty electric vans completed 61,000 kilometres over six months with 97 percent route completion. "
         "Energy cost per kilometre was 58 percent lower than diesel, and maintenance needs were lower than "
         "expected. Drivers reported that range was sufficient for urban routes but not for rural routes longer "
         "than 180 kilometres. Two depot chargers failed and were repaired under warranty."),
        ("Decisions and Next Steps",
         "The group agreed to replace 60 diesel vans in Amsterdam and Rotterdam with electric models during 2027, "
         "subject to board approval of the capital budget. Facilities will install 40 additional charging points. "
         "Finance will prepare a total cost of ownership comparison by the end of October."),
    ]),
    "api_integration_guide.docx": ("Shipping API Integration Guide", [
        ("Authentication",
         "Merchants connect to the Brightfield shipping API using OAuth 2.0 client credentials. Request an access "
         "token from the /oauth/token endpoint with your client ID and secret; tokens expire after 60 minutes. "
         "Keep your client secret on your server and never embed it in a mobile app or website."),
        ("Creating Shipments",
         "Send a POST request to /v2/shipments with the recipient address, parcel weight and dimensions and the "
         "requested service level. The response contains a tracking number and a link to the shipping label in "
         "PDF format."),
        ("Rate Limits and Webhooks",
         "Each merchant account may send up to 100 requests per minute. Requests above the limit return HTTP "
         "status 429 and should be retried with exponential backoff. Subscribe to webhooks to receive status "
         "events such as picked up, out for delivery and delivered instead of polling the tracking endpoint."),
    ]),
}

# ---------------------------------------------------------------------------
# PPTX documents
# ---------------------------------------------------------------------------

PPTXS = {
    "sales_kickoff_2026.pptx": [
        ("Sales Kickoff 2026", ["Theme: grow with our existing customers",
                                "Agenda: results, targets, priorities and recognition"]),
        ("2026 Results", ["New annual contract value of 21.5M EUR against a target of 20M EUR",
                          "Win rate improved from 24% to 29%",
                          "Biggest wins: two national fashion retailers"]),
        ("2027 Targets by Region", ["North: 9M EUR new business", "South: 7M EUR new business",
                                    "Benelux metro same-day: 5M EUR new business",
                                    "Upsell: 15% of existing accounts adopt the returns service"]),
        ("Priorities", ["Cross-sell returns management and warehousing",
                        "Shorten the sales cycle with standard contract templates",
                        "Every account manager holds quarterly business reviews"]),
    ],
    "customer_churn_analysis.pptx": [
        ("Customer Churn Analysis H1 2026", ["Scope: 1,840 business customers with active contracts"]),
        ("Key Findings", ["Annual churn rate rose to 13.5% from 11.2%",
                          "62% of lost customers had two or more late deliveries in their final quarter",
                          "Customers without a dedicated account manager churn twice as often"]),
        ("Churn Drivers", ["Late or missed deliveries", "Billing errors and unclear surcharges",
                           "Competitor price undercutting in the small merchant segment"]),
        ("Retention Actions", ["Early warning score using delivery and support ticket data",
                               "Proactive calls to at-risk accounts",
                               "Loyalty discount for three-year renewals"]),
    ],
    "sustainability_update_2026.pptx": [
        ("Sustainability Update 2026", ["Our commitment: net zero operations by 2040"]),
        ("Emissions", ["Scope 1 and 2 CO2 emissions down 14% versus the 2024 baseline",
                       "Electric vans now make up 12% of the urban fleet",
                       "Solar panels installed on two warehouse roofs"]),
        ("Packaging and Waste", ["Plastic void fill replaced with recycled paper",
                                 "Recycling rate at depots reached 78%",
                                 "Reusable totes trial with grocery customers"]),
        ("Next Steps", ["Submit a science-based emissions target in 2027",
                        "Add supplier sustainability scoring to procurement"]),
    ],
    "cybersecurity_awareness_training.pptx": [
        ("Cybersecurity Awareness Training", ["Mandatory for all staff every year", "Duration: 45 minutes"]),
        ("Spotting Phishing", ["Check the sender address carefully",
                               "Be suspicious of urgency, threats or unexpected invoices",
                               "Hover over links before clicking",
                               "When in doubt, use the Report Phishing button"]),
        ("Protecting Your Accounts", ["Use the password manager to create long unique passwords",
                                      "Approve MFA prompts only when you are signing in yourself",
                                      "Lock your screen when you leave your desk"]),
        ("Social Engineering at the Depot", ["Do not let people follow you through secure doors",
                                             "Verify callers who ask for delivery addresses or customer details"]),
    ],
    "product_roadmap_2027.pptx": [
        ("Product Roadmap 2027", ["Customer portal, merchant API and driver app"]),
        ("Q1 and Q2", ["Dynamic route optimisation using live traffic data", "Driver app offline mode",
                       "Self-service returns labels in the customer portal"]),
        ("Q3 and Q4", ["Delivery time predictions shown to recipients", "Merchant analytics dashboard",
                       "API v3 with bulk shipment creation"]),
        ("Dependencies", ["Data platform migration completed by March", "Two additional mobile developers hired"]),
    ],
    "hiring_plan_2027.pptx": [
        ("Hiring Plan 2027", ["Total planned headcount growth: 85 roles"]),
        ("Roles by Department", ["Operations: 50 warehouse operatives and 12 drivers",
                                 "Technology: 8 engineers including 2 mobile developers",
                                 "Sales: 6 account managers", "Finance and HR: 9 roles"]),
        ("Recruiting Approach", ["Employee referral bonus of 750 EUR",
                                 "Partnerships with vocational schools for warehouse roles",
                                 "Target time to hire: 35 days"]),
        ("Budget", ["Recruitment budget of 420,000 EUR",
                    "Agency use limited to specialist technology roles"]),
    ],
    "board_update_q3_2026.pptx": [
        ("Board Update Q3 2026", ["Prepared for the Board of Directors, October 2026"]),
        ("Financial Highlights", ["Revenue 48.2M EUR, up 11% year on year", "Operating margin 8.5%",
                                  "Net debt reduced to 18M EUR"]),
        ("Operational Highlights", ["On-time delivery 94.1% against a target of 95%",
                                    "Rotterdam cross-dock opened in July",
                                    "Tracking portal outage in August; corrective actions completed"]),
        ("Decisions Requested", ["Approve the 2027 capital budget for 60 electric vans",
                                 "Approve expansion of same-day delivery to Eindhoven"]),
    ],
    "delivery_operations_review.pptx": [
        ("Delivery Operations Review September 2026", []),
        ("Performance", ["On-time delivery rate 94.1%, down from 95.3% in June",
                         "First-attempt delivery success 89%", "Average parcels per route: 142"]),
        ("Causes of Delays", ["Road works around the Utrecht ring road",
                              "Driver shortages during school holidays",
                              "Late linehaul arrivals from the Rotterdam hub"]),
        ("Improvement Actions", ["Earlier depot cut-off times", "Temporary agency drivers for peak season",
                                 "Re-sequencing of routes with the new optimisation tool"]),
    ],
}

# ---------------------------------------------------------------------------
# CSV documents
# ---------------------------------------------------------------------------

CSVS = {
    "support_tickets.csv": [
        {"ticket_id": "T-1001", "category": "Delivery", "customer_issue": "Parcel marked as delivered but customer did not receive it", "resolution": "Driver GPS checked, parcel found with a neighbour"},
        {"ticket_id": "T-1002", "category": "Billing", "customer_issue": "Charged twice for the same shipment", "resolution": "Duplicate charge refunded"},
        {"ticket_id": "T-1003", "category": "Damaged goods", "customer_issue": "Glass vase arrived broken", "resolution": "Claim approved and refund of item value issued"},
        {"ticket_id": "T-1004", "category": "Account", "customer_issue": "Cannot log in to merchant portal after password reset", "resolution": "MFA device re-registered"},
        {"ticket_id": "T-1005", "category": "Delivery", "customer_issue": "Delivery window missed for the third time this month", "resolution": "Escalated to depot manager, priority routing applied"},
        {"ticket_id": "T-1006", "category": "API", "customer_issue": "Shipment creation returns error 429", "resolution": "Merchant advised to implement backoff, rate limit explained"},
        {"ticket_id": "T-1007", "category": "Returns", "customer_issue": "Customer wants to return an order but has no printer for the label", "resolution": "QR code return label sent by email"},
        {"ticket_id": "T-1008", "category": "Billing", "customer_issue": "Unclear fuel surcharge on invoice", "resolution": "Surcharge calculation explained, credit note issued"},
        {"ticket_id": "T-1009", "category": "Delivery", "customer_issue": "Address change requested after dispatch", "resolution": "Parcel redirected to a pickup point"},
        {"ticket_id": "T-1010", "category": "Tracking", "customer_issue": "Tracking page not loading", "resolution": "Linked to tracking portal outage on 14 August"},
        {"ticket_id": "T-1011", "category": "Damaged goods", "customer_issue": "Box crushed and contents wet", "resolution": "Photos collected, claim forwarded to insurer"},
        {"ticket_id": "T-1012", "category": "Account", "customer_issue": "Request to delete my personal data", "resolution": "Forwarded to the Data Protection Officer"},
    ],
    "service_catalog.csv": [
        {"service": "Standard Parcel", "description": "Nationwide delivery for parcels up to 30 kg", "price_eur": "6.50", "delivery_time": "1-2 working days"},
        {"service": "Same-Day Metro", "description": "Same-day delivery in Amsterdam, Rotterdam and Utrecht for orders placed before noon", "price_eur": "12.90", "delivery_time": "Same evening"},
        {"service": "Next-Day Express", "description": "Guaranteed next-day delivery before 12:00", "price_eur": "9.80", "delivery_time": "Next working day"},
        {"service": "Freight Pallet", "description": "Pallet transport up to 1,000 kg with tail-lift truck", "price_eur": "85.00", "delivery_time": "2-3 working days"},
        {"service": "Returns Management", "description": "Prepaid return labels and processing of returned goods", "price_eur": "4.20", "delivery_time": "2-4 working days"},
        {"service": "Warehousing", "description": "Storage and pick-and-pack fulfilment, price per pallet per month", "price_eur": "28.00", "delivery_time": "Not applicable"},
        {"service": "Cold Chain", "description": "Temperature-controlled delivery for food and pharmaceuticals", "price_eur": "18.50", "delivery_time": "Next working day"},
        {"service": "International Economy", "description": "Delivery to EU countries through our partner network", "price_eur": "14.00", "delivery_time": "3-6 working days"},
    ],
    "it_asset_inventory.csv": [
        {"asset_tag": "IT-2041", "device_type": "Laptop", "model": "Business laptop 14-inch", "department": "Finance", "purchase_date": "2022-03-14", "warranty_end": "2025-03-13", "status": "In use, warranty expired"},
        {"asset_tag": "IT-2042", "device_type": "Laptop", "model": "Business laptop 14-inch", "department": "HR", "purchase_date": "2022-03-14", "warranty_end": "2025-03-13", "status": "Scheduled for replacement"},
        {"asset_tag": "IT-2310", "device_type": "Laptop", "model": "Business laptop 15-inch", "department": "Sales", "purchase_date": "2024-06-02", "warranty_end": "2027-06-01", "status": "In use"},
        {"asset_tag": "IT-2311", "device_type": "Laptop", "model": "Developer workstation laptop", "department": "Technology", "purchase_date": "2025-01-20", "warranty_end": "2028-01-19", "status": "In use"},
        {"asset_tag": "IT-3105", "device_type": "Handheld scanner", "model": "Rugged barcode scanner", "department": "Operations", "purchase_date": "2023-09-11", "warranty_end": "2026-09-10", "status": "In use, warranty expired"},
        {"asset_tag": "IT-3106", "device_type": "Handheld scanner", "model": "Rugged barcode scanner", "department": "Operations", "purchase_date": "2023-09-11", "warranty_end": "2026-09-10", "status": "Under repair"},
        {"asset_tag": "IT-4020", "device_type": "Label printer", "model": "Thermal label printer", "department": "Operations", "purchase_date": "2024-02-05", "warranty_end": "2027-02-04", "status": "In use"},
        {"asset_tag": "IT-5001", "device_type": "Mobile phone", "model": "Company smartphone", "department": "Drivers", "purchase_date": "2025-04-15", "warranty_end": "2027-04-14", "status": "In use"},
        {"asset_tag": "IT-5002", "device_type": "Mobile phone", "model": "Company smartphone", "department": "Drivers", "purchase_date": "2021-11-03", "warranty_end": "2023-11-02", "status": "Lost, remotely wiped"},
        {"asset_tag": "IT-6010", "device_type": "Docking station", "model": "USB-C dock", "department": "Head Office", "purchase_date": "2024-06-02", "warranty_end": "2026-06-01", "status": "In use"},
    ],
    "training_courses.csv": [
        {"course_code": "SEC-101", "title": "Cybersecurity Awareness", "description": "Recognise phishing and protect your accounts", "duration_hours": "0.75", "mandatory_for": "All staff"},
        {"course_code": "FLT-200", "title": "Forklift Operator Certification", "description": "Practical and theory training for counterbalance forklifts and reach trucks", "duration_hours": "16", "mandatory_for": "Warehouse operatives"},
        {"course_code": "FA-110", "title": "First Aid at Work", "description": "Emergency first aid and use of defibrillators", "duration_hours": "6", "mandatory_for": "Shift leaders"},
        {"course_code": "GDPR-120", "title": "Data Protection Essentials", "description": "Handling personal data and subject access requests", "duration_hours": "1", "mandatory_for": "All office staff"},
        {"course_code": "DRV-300", "title": "Defensive Driving", "description": "Safe and fuel-efficient driving for van drivers", "duration_hours": "4", "mandatory_for": "Drivers"},
        {"course_code": "EV-310", "title": "Electric Van Familiarisation", "description": "Charging procedures, range management and regenerative braking", "duration_hours": "2", "mandatory_for": "Drivers"},
        {"course_code": "LDR-400", "title": "First-Time Manager Programme", "description": "Coaching, feedback and performance conversations", "duration_hours": "12", "mandatory_for": "New managers"},
        {"course_code": "CS-150", "title": "Customer Service Excellence", "description": "De-escalation and complaint handling", "duration_hours": "3", "mandatory_for": "Support agents"},
        {"course_code": "MH-130", "title": "Manual Handling", "description": "Safe lifting techniques for warehouse work", "duration_hours": "1.5", "mandatory_for": "Warehouse operatives"},
    ],
    "fleet_vehicles.csv": [
        {"vehicle_id": "V-101", "vehicle_type": "Panel van 3.5t", "fuel_type": "Diesel", "depot": "Amsterdam", "year": "2021", "odometer_km": "148200", "next_service_due": "2026-12-03", "notes": ""},
        {"vehicle_id": "V-102", "vehicle_type": "Panel van 3.5t", "fuel_type": "Diesel", "depot": "Amsterdam", "year": "2022", "odometer_km": "112950", "next_service_due": "2027-01-15", "notes": ""},
        {"vehicle_id": "V-103", "vehicle_type": "Panel van 3.5t", "fuel_type": "Electric", "depot": "Utrecht", "year": "2026", "odometer_km": "31400", "next_service_due": "2027-03-10", "notes": "Pilot vehicle, 220 km range"},
        {"vehicle_id": "V-104", "vehicle_type": "Panel van 3.5t", "fuel_type": "Electric", "depot": "Utrecht", "year": "2026", "odometer_km": "29870", "next_service_due": "2027-03-10", "notes": "Pilot vehicle, 220 km range"},
        {"vehicle_id": "V-105", "vehicle_type": "Cargo bike", "fuel_type": "Electric", "depot": "Amsterdam", "year": "2025", "odometer_km": "8100", "next_service_due": "2026-11-30", "notes": "City centre last-mile deliveries"},
        {"vehicle_id": "V-106", "vehicle_type": "Box truck 7.5t", "fuel_type": "Diesel", "depot": "Rotterdam", "year": "2020", "odometer_km": "231400", "next_service_due": "2026-11-20", "notes": "Tail lift"},
        {"vehicle_id": "V-107", "vehicle_type": "Box truck 7.5t", "fuel_type": "HVO biofuel", "depot": "Rotterdam", "year": "2023", "odometer_km": "96300", "next_service_due": "2027-02-02", "notes": ""},
        {"vehicle_id": "V-108", "vehicle_type": "Refrigerated van", "fuel_type": "Diesel", "depot": "Eindhoven", "year": "2022", "odometer_km": "104700", "next_service_due": "2026-12-18", "notes": "Cold chain deliveries"},
        {"vehicle_id": "V-109", "vehicle_type": "Tractor unit", "fuel_type": "Diesel", "depot": "Rotterdam", "year": "2021", "odometer_km": "402100", "next_service_due": "2026-12-01", "notes": "Linehaul between hubs"},
        {"vehicle_id": "V-110", "vehicle_type": "Panel van 3.5t", "fuel_type": "Electric", "depot": "Rotterdam", "year": "2026", "odometer_km": "12050", "next_service_due": "2027-05-22", "notes": ""},
    ],
    "monthly_revenue_by_region.csv": [
        {"month": "2026-07", "region": "North", "revenue_eur": "6050000", "parcels_delivered": "812000", "on_time_delivery_rate": "95.0%"},
        {"month": "2026-07", "region": "South", "revenue_eur": "4950000", "parcels_delivered": "671000", "on_time_delivery_rate": "94.6%"},
        {"month": "2026-07", "region": "Metro", "revenue_eur": "4800000", "parcels_delivered": "702000", "on_time_delivery_rate": "93.8%"},
        {"month": "2026-08", "region": "North", "revenue_eur": "6100000", "parcels_delivered": "820000", "on_time_delivery_rate": "94.7%"},
        {"month": "2026-08", "region": "South", "revenue_eur": "5000000", "parcels_delivered": "680000", "on_time_delivery_rate": "94.2%"},
        {"month": "2026-08", "region": "Metro", "revenue_eur": "4950000", "parcels_delivered": "731000", "on_time_delivery_rate": "93.1%"},
        {"month": "2026-09", "region": "North", "revenue_eur": "6200000", "parcels_delivered": "834000", "on_time_delivery_rate": "94.5%"},
        {"month": "2026-09", "region": "South", "revenue_eur": "5050000", "parcels_delivered": "688000", "on_time_delivery_rate": "94.0%"},
        {"month": "2026-09", "region": "Metro", "revenue_eur": "5100000", "parcels_delivered": "755000", "on_time_delivery_rate": "93.6%"},
    ],
    "supplier_register.csv": [
        {"supplier": "Swiftline IT Services", "category": "IT services", "services_provided": "Managed hosting and 24/7 support for the warehouse management system and tracking portal", "contract_end": "2028-01-31", "annual_spend_eur": "620000", "risk_rating": "Medium"},
        {"supplier": "GreenPack BV", "category": "Packaging", "services_provided": "Recycled cardboard boxes and paper void fill", "contract_end": "2027-06-30", "annual_spend_eur": "410000", "risk_rating": "Low"},
        {"supplier": "Voltway Charging", "category": "Energy infrastructure", "services_provided": "Installation and maintenance of electric vehicle charging points at depots", "contract_end": "2029-03-31", "annual_spend_eur": "275000", "risk_rating": "Medium"},
        {"supplier": "FuelCard Europe", "category": "Fuel", "services_provided": "Fuel cards for diesel and HVO for the van and truck fleet", "contract_end": "2027-12-31", "annual_spend_eur": "3100000", "risk_rating": "Low"},
        {"supplier": "SafeStep Workwear", "category": "Workwear and PPE", "services_provided": "Safety boots, high-visibility vests and cut-resistant gloves", "contract_end": "2027-09-30", "annual_spend_eur": "96000", "risk_rating": "Low"},
        {"supplier": "TempStaff Agency", "category": "Temporary labour", "services_provided": "Agency warehouse operatives and drivers during peak season", "contract_end": "2027-01-31", "annual_spend_eur": "1850000", "risk_rating": "High"},
        {"supplier": "Northern Mutual Insurance", "category": "Insurance", "services_provided": "Cargo, fleet and public liability insurance", "contract_end": "2026-12-31", "annual_spend_eur": "720000", "risk_rating": "Medium"},
        {"supplier": "OfficeHub", "category": "Office supplies", "services_provided": "Stationery, printer paper and toner for all offices", "contract_end": "2027-04-30", "annual_spend_eur": "38000", "risk_rating": "Low"},
    ],
    "office_locations.csv": [
        {"site": "Head Office", "city": "Amsterdam", "site_type": "Office", "employees": "210", "opened": "2012", "facilities": "Meeting rooms, showers, bike storage, wellbeing room"},
        {"site": "Amsterdam West Depot", "city": "Amsterdam", "site_type": "Depot", "employees": "145", "opened": "2015", "facilities": "Sorting hall, 30 loading docks, EV chargers"},
        {"site": "Rotterdam Hub", "city": "Rotterdam", "site_type": "Sorting hub", "employees": "320", "opened": "2010", "facilities": "Automated sorter, 60 loading docks, customs office"},
        {"site": "Rotterdam Cross-Dock", "city": "Rotterdam", "site_type": "Cross-dock", "employees": "85", "opened": "2026", "facilities": "24 loading docks, opened July 2026"},
        {"site": "Utrecht Depot", "city": "Utrecht", "site_type": "Depot", "employees": "120", "opened": "2017", "facilities": "Electric van pilot site, 20 charging points"},
        {"site": "Eindhoven Depot", "city": "Eindhoven", "site_type": "Depot", "employees": "95", "opened": "2019", "facilities": "Cold chain room, 18 loading docks"},
        {"site": "Groningen Depot", "city": "Groningen", "site_type": "Depot", "employees": "60", "opened": "2021", "facilities": "12 loading docks"},
        {"site": "Customer Service Centre", "city": "Utrecht", "site_type": "Office", "employees": "70", "opened": "2018", "facilities": "Contact centre, training room"},
    ],
}

# ---------------------------------------------------------------------------
# Queries and relevance labels (document level, binary)
#
# Labelling rule: a document is relevant if a person asking the query would
# want to open it because it substantially answers the question. A passing
# mention of a word is not enough. Queries of type "no_answer" have no
# relevant document in the corpus.
# ---------------------------------------------------------------------------

QUERIES = [
    # --- exact keyword queries: wording taken from the documents ---
    ("Q01", "keyword", "paternity leave", ["employee_leave_policy.pdf"]),
    ("Q02", "keyword", "multi-factor authentication", ["information_security_policy.pdf", "cybersecurity_awareness_training.pptx"]),
    ("Q03", "keyword", "forklift certification", ["warehouse_safety_manual.pdf", "training_courses.csv"]),
    ("Q04", "keyword", "per diem meal allowance", ["travel_expense_policy.pdf"]),
    ("Q05", "keyword", "API rate limits", ["api_integration_guide.docx"]),
    ("Q06", "keyword", "Q3 2026 revenue and operating margin", ["q3_2026_financial_report.pdf", "board_update_q3_2026.pptx"]),
    ("Q07", "keyword", "purchase order approval thresholds", ["procurement_procedure.docx"]),
    ("Q08", "keyword", "customer churn rate", ["customer_churn_analysis.pptx"]),
    ("Q09", "keyword", "laptop warranty end date", ["it_asset_inventory.csv"]),
    ("Q10", "keyword", "99.9 percent availability service credits", ["vendor_contract_summary.pdf"]),
    ("Q11", "keyword", "CO2 emissions", ["sustainability_update_2026.pptx"]),
    # --- semantic paraphrases: natural questions using different words ---
    ("Q12", "paraphrase", "how much time off do new dads get when a baby is born", ["employee_leave_policy.pdf"]),
    ("Q13", "paraphrase", "I got a weird email asking for my login details, what should I do", ["information_security_policy.pdf", "cybersecurity_awareness_training.pptx"]),
    ("Q14", "paraphrase", "getting my money back after paying for accommodation on a work trip", ["travel_expense_policy.pdf"]),
    ("Q15", "paraphrase", "why did customers lose access to package tracking", ["incident_postmortem_tracking_outage.docx"]),
    ("Q16", "paraphrase", "what are we doing to reduce our impact on the planet", ["sustainability_update_2026.pptx"]),
    ("Q17", "paraphrase", "which delivery vehicles are battery powered", ["fleet_vehicles.csv", "meeting_minutes_fleet_electrification.docx"]),
    ("Q18", "paraphrase", "how long do we hold on to client information before throwing it away", ["data_retention_policy.pdf"]),
    ("Q19", "paraphrase", "what will happen during my first days at the company", ["new_hire_onboarding_checklist.docx"]),
    ("Q20", "paraphrase", "how is it decided whether I get a salary raise", ["performance_review_process.docx"]),
    ("Q21", "paraphrase", "can I do my job from my house some days", ["remote_work_guidelines.pdf"]),
    ("Q22", "paraphrase", "which customers might leave us and why", ["customer_churn_analysis.pptx"]),
    ("Q23", "paraphrase", "how much money did we bring in between July and September", ["q3_2026_financial_report.pdf", "board_update_q3_2026.pptx", "monthly_revenue_by_region.csv"]),
    ("Q24", "paraphrase", "how can an online shop plug its systems into ours", ["api_integration_guide.docx"]),
    ("Q25", "paraphrase", "what protective clothing must warehouse workers wear", ["warehouse_safety_manual.pdf"]),
    ("Q26", "paraphrase", "an annoyed client wants money back, how should I respond", ["customer_support_playbook.docx"]),
    ("Q27", "paraphrase", "can you transport medicine that has to stay cold", ["service_catalog.csv"]),
    ("Q28", "paraphrase", "why are parcels arriving late", ["delivery_operations_review.pptx"]),
    ("Q29", "paraphrase", "what new features is the tech team building next year", ["product_roadmap_2027.pptx"]),
    ("Q30", "paraphrase", "how many people are we going to recruit", ["hiring_plan_2027.pptx"]),
    ("Q31", "paraphrase", "under what conditions can we end the agreement with our hosting provider", ["vendor_contract_summary.pdf"]),
    # --- ambiguous / broad one- or two-word queries with several relevant files ---
    ("Q32", "ambiguous", "security", ["information_security_policy.pdf", "cybersecurity_awareness_training.pptx"]),
    ("Q33", "ambiguous", "training", ["training_courses.csv", "cybersecurity_awareness_training.pptx"]),
    ("Q34", "ambiguous", "contract", ["vendor_contract_summary.pdf", "supplier_register.csv"]),
    ("Q35", "ambiguous", "delivery performance", ["delivery_operations_review.pptx", "board_update_q3_2026.pptx", "monthly_revenue_by_region.csv"]),
    ("Q36", "ambiguous", "customer data", ["data_retention_policy.pdf", "information_security_policy.pdf"]),
    ("Q37", "ambiguous", "refunds", ["customer_support_playbook.docx", "support_tickets.csv"]),
    ("Q38", "ambiguous", "safety", ["warehouse_safety_manual.pdf", "training_courses.csv"]),
    ("Q39", "ambiguous", "returns", ["service_catalog.csv", "support_tickets.csv", "product_roadmap_2027.pptx", "sales_kickoff_2026.pptx"]),
    # --- no relevant document exists in the corpus ---
    ("Q40", "no_answer", "stock option vesting schedule", []),
    ("Q41", "no_answer", "where is the company christmas party", []),
    ("Q42", "no_answer", "how do I file a patent for an invention", []),
    ("Q43", "no_answer", "quantum computing research projects", []),
    ("Q44", "no_answer", "visa sponsorship for foreign workers", []),
]


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (title, pages) in PDFS.items():
        write_pdf(DOCS_DIR / name, title, pages)
    for name, (title, sections) in DOCXS.items():
        write_docx(DOCS_DIR / name, title, sections)
    for name, slides in PPTXS.items():
        write_pptx(DOCS_DIR / name, slides)
    for name, rows in CSVS.items():
        write_csv(DOCS_DIR / name, rows)

    all_files = set(PDFS) | set(DOCXS) | set(PPTXS) | set(CSVS)
    labels = []
    for query_id, _, _, relevant in QUERIES:
        for file_name in relevant:
            assert file_name in all_files, f"{query_id}: unknown file {file_name}"
            labels.append({"query_id": query_id, "file_name": file_name, "relevance": 1})

    pd.DataFrame([{"query_id": q, "query_type": t, "query": text} for q, t, text, _ in QUERIES]) \
        .to_csv(DATA_DIR / "queries.csv", index=False)
    pd.DataFrame(labels).to_csv(DATA_DIR / "relevance_labels.csv", index=False)

    print(f"Wrote {len(all_files)} documents to {DOCS_DIR}")
    print(f"Wrote {len(QUERIES)} queries and {len(labels)} relevance labels to {DATA_DIR}")


if __name__ == "__main__":
    main()
