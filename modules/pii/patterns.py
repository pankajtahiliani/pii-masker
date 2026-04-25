"""
PII detection patterns — regex, context labels, name rules, mask labels.
Covers IN, AU, UK, US, CA, EU, UAE/GCC, SG + universal patterns.
"""
import re

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL PII PATTERN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

REGEX_PATTERNS = {
    # ── Universal ────────────────────────────────────────────────────────────
    "email":        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b',
    "url":          r'https?://[^\s<>"]+|www\.[^\s<>"]+',
    "ip_address":   r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "ipv6":         r'\b([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
    "credit_card":  r'\b(?:\d{4}[\s\-]?){3}\d{4}\b',
    "gender":       r'\b(?:Male|Female|Other|Non-binary|Transgender|Prefer not to say)\b',
    "dob":          r'\b(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b',

    # Strong password pattern (mixed case + digit + symbol, 8+ chars)
    "password":     r'\b(?=[A-Za-z0-9!@#$%^&*()\-_=+{};:,<.>]*[A-Z])(?=[A-Za-z0-9!@#$%^&*()\-_=+{};:,<.>]*[a-z])(?=[A-Za-z0-9!@#$%^&*()\-_=+{};:,<.>]*\d)(?=[A-Za-z0-9!@#$%^&*()\-_=+{};:,<.>]*[!@#$%^&*()\-_=+{};:,<.>])[A-Za-z0-9!@#$%^&*()\-_=+{};:,<.>]{8,}\b',

    # Tech credentials
    "aws_key":      r'\b(?:AKIA|ASIA|AROA)[A-Z0-9]{16}\b',
    "aws_secret":   r'\b[A-Za-z0-9/+=]{40}\b',
    "api_key":      r'\b(?:SG\.|rzp_(?:test|live)_|sk_(?:test|live)_|pk_(?:test|live)_|AC[a-f0-9]{30,}|AIzaSy|gh[pousr]_)[A-Za-z0-9\._\-]{10,}\b',
    "jwt_token":    r'\beyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b',

    # System usernames
    "admin_user":   r'\b(?:admin|prod|uat|dev|root|superuser|sysadmin)_[a-z0-9_]+\b',
    "username":     r'\b[a-z][a-z0-9]*_[a-z][a-z0-9]*(?:_[a-z0-9]+)?\b',

    # ── INDIA ────────────────────────────────────────────────────────────────
    "in_phone":     r'(?:\+91|0091|91)[\s\-]?[6-9]\d{4}[\s\-]?\d{5}\b',
    "in_aadhaar":   r'\b\d{4}[\s\-]\d{4}[\s\-]\d{4}\b',
    "in_pan":       r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
    "in_gst":       r'\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b',
    "in_passport":  r'\b[A-Z][1-9]\d{7}\b',
    "in_voter_id":  r'\b[A-Z]{3}[0-9]{7}\b',
    "in_driving":   r'\b[A-Z]{2}\d{2}\s?\d{11}\b',
    "in_pincode":   r'\b[1-9][0-9]{5}\b',
    "in_ifsc":      r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
    "in_upi":       r'\b[\w.\-]+@(?:okicici|oksbi|okaxis|okhdfcbank|paytm|ybl|ibl|axl|upi)\b',

    # ── AUSTRALIA ────────────────────────────────────────────────────────────
    "au_phone":     r'(?:\+61|0061)?[\s\-]?(?:0?[2378]\d{8}|4\d{8})\b',
    "au_tfn":       r'\b\d{3}[\s\-]?\d{3}[\s\-]?\d{3}\b',
    "au_abn":       r'\b\d{2}[\s]?\d{3}[\s]?\d{3}[\s]?\d{3}\b',
    "au_acn":       r'\b\d{3}[\s]?\d{3}[\s]?\d{3}\b',
    "au_medicare":  r'\b\d{4}[\s]?\d{5}[\s]?\d{1}\b',
    "au_postcode":  r'\b(?:0[89]\d{2}|[1-9]\d{3})\b',
    "au_passport":  r'\b[A-Z]{1,2}\d{7}\b',
    "au_drivers":   r'\b\d{8,9}\b',

    # ── UNITED KINGDOM ───────────────────────────────────────────────────────
    "uk_phone":     r'(?:\+44|0044)?[\s\-]?(?:0?[1-9]\d{9}|7\d{9})\b',
    "uk_nino":      r'\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b',
    "uk_postcode":  r'\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b',
    "uk_passport":  r'\b\d{9}\b',
    "uk_sort_code": r'\b\d{2}[\s\-]\d{2}[\s\-]\d{2}\b',
    "uk_bank_acct": r'\b\d{8}\b',
    "uk_nhs":       r'\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b',
    "uk_utr":       r'\b\d{10}\b',
    "uk_company":   r'\b(?:SC|NI|OC|SO|NC)?\d{6,8}\b',

    # ── UNITED STATES ────────────────────────────────────────────────────────
    "us_phone":     r'(?:\+1[\s\-]?)?(?:\(\d{3}\)[\s\-]?|\d{3}[\s\-])\d{3}[\s\-]\d{4}\b',
    "us_ssn":       r'\b(?!000|666|9\d{2})\d{3}[\s\-]\d{2}[\s\-]\d{4}\b',
    "us_zip":       r'\b\d{5}(?:\-\d{4})?\b',
    "us_passport":  r'\b[A-Z]\d{8}\b',
    "us_ein":       r'\b\d{2}\-\d{7}\b',
    "us_itin":      r'\b9\d{2}[\s\-]\d{2}[\s\-]\d{4}\b',
    "us_drivers":   r'\b[A-Z]\d{7}\b',
    "us_dea":       r'\b[A-Z]{2}\d{7}\b',
    "us_npi":       r'\b\d{10}\b',

    # ── CANADA ───────────────────────────────────────────────────────────────
    "ca_phone":     r'(?:\+1[\s\-]?)?(?:\(\d{3}\)[\s\-]?|\d{3}[\s\-])\d{3}[\s\-]\d{4}\b',
    "ca_sin":       r'\b\d{3}[\s\-]\d{3}[\s\-]\d{3}\b',
    "ca_postcode":  r'\b[A-Z]\d[A-Z][\s]?\d[A-Z]\d\b',
    "ca_passport":  r'\b[A-Z]{2}\d{6}\b',
    "ca_health":    r'\b\d{10}\b',

    # ── EUROPEAN UNION (generic + major countries) ────────────────────────────
    "eu_phone":     r'(?:\+(?:33|49|34|39|31|32|41|43|45|46|47|48|351|353|358|420|421|36|40|359|370|371|372|386|385|356|357)[\s\-]?)[\d\s\-\(\)]{7,15}\b',
    "eu_vat":       r'\b(?:GB|FR|DE|IT|ES|NL|BE|PL|SE|AT|DK|FI|PT|IE|CZ|SK|HU|RO|BG|HR|LT|LV|EE|SI|CY|MT|LU)\s?[A-Z0-9]{8,12}\b',
    "de_id":        r'\b[A-Z0-9]{9}\b',
    "fr_nir":       r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b',
    "eu_iban":      r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})\b',

    # ── UAE / GCC ─────────────────────────────────────────────────────────────
    "ae_phone":     r'(?:\+971|00971)?[\s\-]?(?:0?[2-9]\d{7}|5[024568]\d{7})\b',
    "ae_eid":       r'\b784[\s\-]?\d{4}[\s\-]?\d{7}[\s\-]?\d{1}\b',
    "ae_trade_lic": r'\b(?:CN|BN|LN|TL)-?\d{4,10}\b',
    "sa_phone":     r'(?:\+966|00966)?[\s\-]?0?[15]\d{8}\b',
    "sa_iqama":     r'\b[12]\d{9}\b',

    # ── SINGAPORE ────────────────────────────────────────────────────────────
    "sg_phone":     r'(?:\+65|0065)?[\s\-]?[689]\d{7}\b',
    "sg_nric":      r'\b[STFGM]\d{7}[A-Z]\b',
    "sg_uen":       r'\b\d{9}[A-Z]\b',
    "sg_passport":  r'\bE\d{7}[A-Z]\b',

    # ── ADDRESSES (global structure) ──────────────────────────────────────────
    "address_unit":   r'\b(?:Apt|Apartment|Suite|Ste|Unit|Flat|Floor|Fl|Room|Rm|Level|Lvl)\.?\s*#?\d+[A-Za-z]?\b',
    "address_street": r'\b\d+\s+[A-Za-z0-9\s]{3,30}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Terrace|Terr|Close|Crescent|Cres|Highway|Hwy|Parade|Circuit|Grove|Gardens?)\b',
    "po_box":         r'\b(?:P\.?O\.?|Post Office)\s*Box\s*\d+\b',
}

# ── Context-aware extraction (keyword: value — works across all regions) ───────
CONTEXT_LABELS = {
    r'(?i)(?:full\s*(?:legal\s*)?name|customer\s*name|contact\s*name|first\s*name|last\s*name|given\s*name|surname|family\s*name)\s*[:\-–]\s*': (r'[^\n,;]{2,60}', 'full_name'),
    r'(?i)(?:gender|sex)\s*[:\-–]\s*':                                   (r'\S+', 'gender'),
    r'(?i)(?:date\s*of\s*birth|dob|birth\s*date)\s*[:\-–]\s*':          (r'[\d\/\-\.A-Za-z\s]{4,20}', 'dob'),
    r'(?i)(?:address|addr|residence|home\s*address|mailing\s*address|street\s*address)\s*[:\-–]\s*': (r'[^\n]{5,120}', 'address'),
    r'(?i)(?:postcode|postal\s*code|zip\s*code?|pin\s*code)\s*[:\-–]\s*': (r'[A-Z0-9\s\-]{3,10}', 'postcode'),
    r'(?i)(?:username|user\s*name|login\s*(?:id|name)|user\s*id|account\s*name)\s*[:\-–]\s*': (r'\S+', 'username'),
    r'(?i)(?:password|passwd|default\s*password|pwd|passphrase)\s*[:\-–]?\s*': (r'\S{4,}', 'password'),
    r'(?i)(?:mobile|cell(?:phone)?|phone|telephone|tel|contact\s*(?:no|number))\s*[:\-–]\s*': (r'[\+\d\s\-\(\)]{7,20}', 'phone'),
    r'(?i)(?:national\s*id|nat\.?\s*id|id\s*(?:number|no)|passport\s*(?:no|number)|driving\s*licen[sc]e)\s*[:\-–]\s*': (r'[A-Z0-9\s\-]{5,20}', 'national_id'),
    r'(?i)(?:ssn|social\s*security)\s*[:\-–]\s*':                        (r'[\d\s\-]{9,11}', 'us_ssn'),
    r'(?i)(?:aadhaar|aadhar|uid)\s*[:\-–]\s*':                          (r'[\d\s\-]{12,14}', 'in_aadhaar'),
    r'(?i)(?:pan\s*(?:number|no)?)\s*[:\-–]\s*':                        (r'[A-Z0-9]{10}', 'in_pan'),
    r'(?i)(?:tfn|tax\s*file\s*(?:number|no))\s*[:\-–]\s*':              (r'[\d\s]{9,11}', 'au_tfn'),
    r'(?i)(?:nino|national\s*insurance)\s*[:\-–]\s*':                    (r'[A-Z0-9\s]{9,11}', 'uk_nino'),
    r'(?i)(?:medicare)\s*[:\-–]\s*':                                     (r'[\d\s]{10,12}', 'au_medicare'),
    r'(?i)(?:access\s*key|secret\s*(?:key|access)|api\s*key|api\s*token)\s*[:\-–]\s*': (r'\S{10,}', 'api_key'),
    r'(?i)(?:bank\s*account|account\s*(?:no|number))\s*[:\-–]\s*':      (r'[\d\s\-]{6,20}', 'bank_account'),
    r'(?i)(?:sort\s*code|routing\s*(?:no|number))\s*[:\-–]\s*':         (r'[\d\s\-]{6,11}', 'sort_code'),
    r'(?i)(?:iban)\s*[:\-–]\s*':                                        (r'[A-Z0-9\s]{15,34}', 'eu_iban'),
    r'(?i)(?:nationality|citizenship|country\s*of\s*origin)\s*[:\-–]\s*': (r'[A-Za-z\s]{3,30}', 'nationality'),
    r'(?i)(?:ethnicity|race)\s*[:\-–]\s*':                              (r'[A-Za-z\s\-]{3,30}', 'ethnicity'),
    r'(?i)(?:religion|faith)\s*[:\-–]\s*':                              (r'[A-Za-z\s]{3,20}', 'religion'),
    r'(?i)(?:salary|annual\s*(?:salary|income|ctc)|compensation|remuneration)\s*[:\-–]\s*': (r'[\d,\.\s\$£€₹]{3,20}', 'salary'),
    r'(?i)(?:dbs|criminal\s*record|background\s*check)\s*[:\-–]\s*':    (r'[A-Za-z0-9\s]{2,30}', 'background_check'),
    r'(?i)(?:disability|medical\s*condition|health\s*condition)\s*[:\-–]\s*': (r'[^\n]{3,60}', 'health'),
}

# ── Proper name detection ─────────────────────────────────────────────────────
NAME_PATTERNS = [
    r'\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Dame|Rev|Hon|Capt|Maj|Col|Gen|Sgt|Fr|Sr|Br)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b',
    r'\b[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20}\b',
    r'\b[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20}\b',
]

NOT_NAMES = {
    'Business','Analyst','Product','Owner','Technical','Architect','Scrum','Master',
    'DevOps','Engineer','Finance','Director','Manager','Senior','Junior','Lead','Head',
    'Chief','Vice','President','Officer','Associate','Consultant','Specialist','Executive',
    'Administrator','Coordinator','Supervisor','Inspector','Controller','Partner',
    'Software','Requirements','Specification','Document','Control','Overview','Background',
    'Objective','Scope','Summary','Revision','History','Contents','Section','Heading',
    'Appendix','Annex','Version','Status','Draft','Final','Initial','Approved','Signed',
    'Access','Level','Default','Password','Username','Login','Logout','Register','Reset',
    'Primary','Secondary','Alternate','Optional','Mandatory','Required','Recommended',
    'Encrypted','Hashed','Masked','Redacted','Restricted','Confidential','Public',
    'Module','System','Platform','Service','Gateway','Bucket','Token','Secret','Key',
    'Environment','Development','Production','Staging','Testing','Integration','Deployment',
    'Backend','Frontend','Database','Server','Client','Infrastructure','Architecture',
    'Application','Interface','Dashboard','Report','Analytics','Pipeline','Workflow',
    'Australia','United','Kingdom','States','America','Canada','Singapore','Emirates',
    'India','Pakistan','Bangladesh','Sri','Lanka','Nepal','Bhutan','Maldives',
    'New','South','Wales','Victoria','Queensland','Western','Northern','Territory',
    'England','Scotland','Wales','Ireland','Northern','London','Manchester','Sydney',
    'Melbourne','Brisbane','Perth','Adelaide','Auckland','Wellington','Christchurch',
    'Delhi','Mumbai','Bengaluru','Chennai','Hyderabad','Kolkata','Jaipur','Noida','Pune',
    'Dubai','Abu','Dhabi','Riyadh','Doha','Kuwait','Bahrain','Muscat',
    'California','Texas','Florida','York','Jersey','Hampshire','Mexico','Orleans',
    'Pradesh','Rajasthan','Maharashtra','Karnataka','Kerala','Gujarat','Bihar',
    'Uttar','Madhya','Himachal','Jammu','Kashmir','Uttarakhand','Jharkhand',
    'Street','Avenue','Road','Boulevard','Lane','Drive','Court','Place','Way',
    'Terrace','Close','Crescent','Highway','Parade','Circuit','Grove','Gardens',
    'Sector','Floor','Plot','Flat','Apartment','Suite','Block','Wing','Tower',
    'Phase','Nagar','Marg','Chowk','Colony','Layout','Extension','Enclave',
    'January','February','March','April','June','July','August','September',
    'October','November','December','Monday','Tuesday','Wednesday','Thursday',
    'Friday','Saturday','Sunday','Today','Tomorrow','Yesterday',
    'Customer','Employee','Team','Register','Legal','Billing','Contract','Reference',
    'Project','Platform','Services','Super','Admin','Backend','Frontend',
    'Third','Party','Payment','Maps','Documents','Sign','Signature','Approval',
    'Review','Change','Full','Name','Mobile','Phone','Role','Gender','Birth',
    'Field','Type','Notes','Format','Generated','Deployment','Token','Service',
    'Male','Female','Other','True','False','None','Null','Yes','No','Not',
    'Available','Applicable','Provided','Contact','Details','Information',
    'Account','Number','Reference','Code','Serial','Batch','Order','Invoice',
}

# ── Fix 1 & 2: Safe value guards ─────────────────────────────────────────────
# Column/field header words — never mask as PII values (Bug Fix 1)
HEADER_WORDS = {
    'Email', 'Date', 'Password', 'Name', 'Phone', 'Mobile', 'Address',
    'Gender', 'Role', 'Status', 'Type', 'Notes', 'Code', 'Number',
    'Username', 'Login', 'Access', 'Level', 'Environment', 'Service',
    'Region', 'Resource', 'Details', 'Summary', 'Version', 'Reference',
    'Signature', 'Designation', 'Location', 'Contact', 'Billing',
    'Development', 'Production', 'UAT', 'Staging',
}

# Date pattern: date-formatted strings are never treated as passwords (Bug Fix 2)
DATE_PATTERN = re.compile(
    r'^\d{1,2}[-\/\.]\w{2,9}[-\/\.]\d{2,4}$'
    r'|\d{4}[-\/\.]\d{1,2}[-\/\.]\d{1,2}'
    r'|\d{1,2}[-\/\.]\d{1,2}[-\/\.]\d{2,4}',
    re.IGNORECASE,
)

# ── High-priority pattern order (Bug Fix 6) ───────────────────────────────────
# Applied BEFORE short numeric patterns to prevent fragmentation.
HIGH_PRIORITY_TYPES = {
    'email', 'url', 'in_phone', 'au_phone', 'uk_phone', 'us_phone',
    'ca_phone', 'eu_phone', 'ae_phone', 'sa_phone', 'sg_phone',
    'api_key', 'aws_key', 'aws_secret', 'jwt_token',
    'in_aadhaar', 'in_gst', 'in_pan', 'uk_nino', 'uk_postcode',
    'us_ssn', 'ca_postcode', 'sg_nric', 'eu_iban', 'eu_vat',
    'credit_card', 'ip_address', 'ipv6',
}

# ── Mask label map ────────────────────────────────────────────────────────────
# Fix: removed duplicate keys (bank_account, sort_code, dob were defined twice).
MASK_LABELS = {
    # Universal
    "email": "[EMAIL]", "url": "[URL]", "ip_address": "[IP]", "ipv6": "[IP]",
    "gender": "[GENDER]", "dob": "[DOB]", "password": "[PASSWORD]",
    "credit_card": "[CREDIT_CARD]", "aws_key": "[AWS_KEY]", "aws_secret": "[AWS_SECRET]",
    "api_key": "[API_KEY]", "jwt_token": "[JWT_TOKEN]",
    "admin_user": "[USERNAME]", "username": "[USERNAME]",
    "full_name": "[FULL_NAME]", "name": "[NAME]", "phone": "[PHONE]",
    "address": "[ADDRESS]", "address_unit": "[ADDRESS]", "address_street": "[ADDRESS]",
    "po_box": "[ADDRESS]", "postcode": "[POSTCODE]", "national_id": "[NATIONAL_ID]",
    "bank_account": "[BANK_ACCOUNT]", "sort_code": "[SORT_CODE]",
    "nationality": "[NATIONALITY]", "ethnicity": "[ETHNICITY]", "religion": "[RELIGION]",
    "salary": "[SALARY]", "background_check": "[BACKGROUND_CHECK]", "health": "[HEALTH]",
    # India
    "in_phone": "[PHONE]", "in_aadhaar": "[AADHAAR]", "in_pan": "[PAN]",
    "in_gst": "[GST]", "in_passport": "[PASSPORT]", "in_voter_id": "[VOTER_ID]",
    "in_driving": "[DRIVING_LICENCE]", "in_pincode": "[POSTCODE]",
    "in_ifsc": "[IFSC_CODE]", "in_upi": "[UPI_ID]",
    # Australia
    "au_phone": "[PHONE]", "au_tfn": "[TAX_FILE_NO]", "au_abn": "[ABN]",
    "au_acn": "[ACN]", "au_medicare": "[MEDICARE]", "au_postcode": "[POSTCODE]",
    "au_passport": "[PASSPORT]", "au_drivers": "[DRIVING_LICENCE]",
    # UK
    "uk_phone": "[PHONE]", "uk_nino": "[NINO]", "uk_postcode": "[POSTCODE]",
    "uk_passport": "[PASSPORT]", "uk_sort_code": "[SORT_CODE]",
    "uk_bank_acct": "[BANK_ACCOUNT]", "uk_nhs": "[NHS_NUMBER]",
    "uk_utr": "[UTR]", "uk_company": "[COMPANY_NO]",
    # US
    "us_phone": "[PHONE]", "us_ssn": "[SSN]", "us_zip": "[ZIP_CODE]",
    "us_passport": "[PASSPORT]", "us_ein": "[EIN]", "us_itin": "[ITIN]",
    "us_drivers": "[DRIVING_LICENCE]", "us_dea": "[DEA_NO]", "us_npi": "[NPI]",
    # Canada
    "ca_phone": "[PHONE]", "ca_sin": "[SIN]", "ca_postcode": "[POSTCODE]",
    "ca_passport": "[PASSPORT]", "ca_health": "[HEALTH_CARD]",
    # EU
    "eu_phone": "[PHONE]", "eu_vat": "[VAT_NO]", "de_id": "[NATIONAL_ID]",
    "fr_nir": "[NATIONAL_ID]", "eu_iban": "[IBAN]",
    # UAE/GCC
    "ae_phone": "[PHONE]", "ae_eid": "[EMIRATES_ID]", "ae_trade_lic": "[TRADE_LICENCE]",
    "sa_phone": "[PHONE]", "sa_iqama": "[IQAMA]",
    # Singapore
    "sg_phone": "[PHONE]", "sg_nric": "[NRIC]", "sg_uen": "[UEN]",
    "sg_passport": "[PASSPORT]",
    # Explicit labels for dynamically generated types (Fix 5)
    "legal_act": "[REDACTED]", "act": "[REDACTED]",
    "date": "[DATE]",
}
