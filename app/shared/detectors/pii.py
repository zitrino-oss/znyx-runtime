import re
from typing import List, Tuple, Dict, Any, Optional, Callable
from app.shared.core.models import DetectorResult, RuleHit, Severity, Decision
from app.shared.core.risk import calculate_risk_score
from app.shared.utils.luhn import luhn_check
from app.shared.utils.checksums import (
    cnpj_check,
    cpf_check,
    mod10_sin_check,
    mod11_nhs_check,
    nif_check,
    verhoeff_check,
)


def _bsn_11_check(number: str) -> bool:
    """Netherlands BSN 11-test: weighted sum of 9 digits (weights 9..2,-1) must be divisible by 11."""
    digits = number.replace(' ', '').replace('-', '')
    if len(digits) != 9 or not digits.isdigit():
        return False
    weights = [9, 8, 7, 6, 5, 4, 3, 2, -1]
    total = sum(int(d) * w for d, w in zip(digits, weights))
    return total % 11 == 0


def _ktp_date_sanity(number: str) -> bool:
    """Indonesian KTP: 16 digits; digits 7-12 encode DD/MM/YY (women add 40 to DD). Sanity-check the date."""
    digits = number.replace(' ', '').replace('-', '')
    if len(digits) != 16 or not digits.isdigit():
        return False
    try:
        day = int(digits[6:8])
        month = int(digits[8:10])
        if day > 40:
            day -= 40
        return 1 <= day <= 31 and 1 <= month <= 12
    except ValueError:
        return False


def _au_tfn_check(number: str) -> bool:
    """Australian TFN mod-11 check (weights 1,4,3,7,5,8,6,9,10 for 9-digit TFNs)."""
    digits = re.sub(r'[\s-]', '', number)
    if len(digits) not in (8, 9) or not digits.isdigit():
        return False
    weights_9 = [1, 4, 3, 7, 5, 8, 6, 9, 10]
    weights_8 = [10, 7, 8, 4, 6, 3, 5, 1]
    weights = weights_9 if len(digits) == 9 else weights_8
    total = sum(int(d) * w for d, w in zip(digits, weights))
    return total % 11 == 0


def _au_abn_check(number: str) -> bool:
    """Australian ABN mod-89 check (subtract 1 from first digit; weighted sum mod 89 must be 0)."""
    digits = re.sub(r'[\s-]', '', number)
    if len(digits) != 11 or not digits.isdigit():
        return False
    weights = [10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    first = int(digits[0]) - 1
    total = first * weights[0] + sum(int(d) * w for d, w in zip(digits[1:], weights[1:]))
    return total % 89 == 0


def _au_acn_check(number: str) -> bool:
    """Australian Company Number mod-10 check (weights 8,7,6,5,4,3,2,1 on the first 8 digits;
    check digit = (10 - sum%10) % 10 equals the 9th digit)."""
    digits = re.sub(r'[\s-]', '', number)
    if len(digits) != 9 or not digits.isdigit():
        return False
    weights = [8, 7, 6, 5, 4, 3, 2, 1]
    total = sum(int(d) * w for d, w in zip(digits[:8], weights))
    check = (10 - (total % 10)) % 10
    return check == int(digits[8])


class PIIDetector:
    """Detects PII (Personally Identifiable Information) in text"""

    # Email pattern (enhanced with Unicode support)
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        re.UNICODE
    )

    # Phone patterns (US formats and international)
    # Requires formatting (separators or parentheses) to avoid matching bare digit
    # sequences like order numbers, product IDs, or timestamps.
    PHONE_PATTERN = re.compile(
        r'(?:\+?1[-.\s]?)'                                        # +1 prefix
        r'\(?([0-9]{3})\)?[-.\s]([0-9]{3})[-.\s]([0-9]{4})\b'    # +1 NXX-NXX-XXXX
        r'|'
        r'\(([0-9]{3})\)[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'   # (NXX) NXX-XXXX
        r'|'
        r'\b([0-9]{3})([-.\s])([0-9]{3})\8([0-9]{4})\b'          # NXX-NXX-XXXX (consistent separator)
    )

    # Contextual keywords that indicate the surrounding number is NOT a phone number
    _PHONE_FALSE_POSITIVE_CONTEXT = re.compile(
        r'(?:order|invoice|ref|reference|confirmation|tracking|id|code|version|'
        r'amount|total|qty|quantity|product|item|sku|part|model|serial|account|'
        r'zip|postal)\s*[#:]?\s*$',
        re.IGNORECASE
    )

    # IPv4 pattern
    IPV4_PATTERN = re.compile(
        r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    )

    # IPv6 pattern (basic matching)
    IPV6_PATTERN = re.compile(
        r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|'  # Full form
        r'(?:[0-9a-fA-F]{1,4}:){1,7}:|'  # Compressed form
        r'::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}|'  # Leading ::
        r'(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}'  # Embedded ::
    )

    # Credit card pattern (13-19 digits, with optional spaces/dashes)
    CARD_PATTERN = re.compile(
        r'\b(?:\d[ -]*?){13,19}\b'
    )

    # SSN pattern (XXX-XX-XXXX format)
    # Requires consistent separators (dashes or spaces) between all groups to avoid
    # matching bare 9-digit numbers like zip+4 codes, order numbers, or account IDs.
    # Intentionally permissive: catches all XXX-XX-XXXX patterns including technically
    # invalid SSNs (000, 666) because guardrails should err on the side of caution.
    SSN_PATTERN = re.compile(
        r'\b(\d{3})([-\s])(\d{2})\2(\d{4})\b'
    )

    # Contextual keywords that indicate a nearby XXX-XX-XXXX is NOT an SSN
    _SSN_FALSE_POSITIVE_CONTEXT = re.compile(
        r'(?:date|born|dob|birthday|expire|expir|issued?|effective|'
        r'zip|postal|order|invoice|ref|reference|phone|fax|ext)\s*[#:]?\s*$',
        re.IGNORECASE
    )

    # API Key patterns
    API_KEY_PATTERNS = [
        # Stripe keys
        (re.compile(r'\b(sk|pk)_(live|test)_[A-Za-z0-9]{24,}\b'), 'stripe', Severity.HIGH),
        # GitHub tokens
        (re.compile(r'\bghp_[A-Za-z0-9]{36,}\b'), 'github', Severity.HIGH),
        (re.compile(r'\bgho_[A-Za-z0-9]{36,}\b'), 'github_oauth', Severity.HIGH),
        # AWS keys
        (re.compile(r'\bAKIA[0-9A-Z]{16}\b'), 'aws', Severity.HIGH),
        # Generic API keys (bearer tokens, etc)
        (re.compile(r'\b(?:api[_-]?key|apikey|api[_-]?token)[=:\s]+["\']?([A-Za-z0-9_\-]{32,})["\']?', re.IGNORECASE), 'generic_api', Severity.HIGH),
        # JWT tokens (basic detection)
        (re.compile(r'\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b'), 'jwt', Severity.MEDIUM),
    ]

    # Passport patterns (US and international)
    PASSPORT_PATTERN = re.compile(
        r'\b[A-Z]{1,2}[0-9]{6,9}\b'  # Common formats: US (9 digits), UK (9 chars), etc.
    )

    # Driver's License patterns (US states - common formats)
    DRIVERS_LICENSE_PATTERN = re.compile(
        r'\b[A-Z]{1,2}[0-9]{5,8}\b|'  # Generic state format
        r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b|'  # SSN-like format used by some states
        r'\b[A-Z][0-9]{3}-[0-9]{3}-[0-9]{2}-[0-9]{3}-[0-9]\b'  # WA format
    )

    # National ID pattern (generic - various countries)
    NATIONAL_ID_PATTERN = re.compile(
        r'\b[A-Z]{2}[0-9]{7,10}[A-Z]?\b'  # Generic format
    )

    # Tax ID patterns (US ITIN and EIN)
    TAX_ID_PATTERN = re.compile(
        r'\b9[0-9]{2}-[0-9]{2}-[0-9]{4}\b|'  # ITIN (9XX-XX-XXXX)
        r'\b[0-9]{2}-[0-9]{7}\b'  # EIN (XX-XXXXXXX)
    )

    # Bank Account patterns
    BANK_ACCOUNT_PATTERN = re.compile(
        r'\b[0-9]{8,17}\b'  # US bank account (8-17 digits)
    )

    # Routing Number pattern (US - 9 digits with checksum)
    ROUTING_NUMBER_PATTERN = re.compile(
        r'\b[0-9]{9}\b'
    )

    # IBAN pattern (International Bank Account Number). IBANs are often
    # written with spaces or dashes every 4 chars for readability, e.g.
    # "GB29 NWBK 6016 1331 9268 19". The old pattern required contiguous
    # alnum, so any formatted IBAN slipped through. Now we allow an optional
    # [\s-] separator between alnum characters in the body. Total alnum body
    # length after the 4-char country+check prefix is 11-30 chars, matching
    # the ISO 13616 per-country bounds.
    IBAN_PATTERN = re.compile(
        r'\b[A-Z]{2}[0-9]{2}(?:[\s-]?[A-Z0-9]){11,30}\b'
    )

    # Physical Address pattern (US format — full with zip)
    ADDRESS_PATTERN = re.compile(
        r'\b\d{1,5}\s+[\w\s]{1,30}(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|parkway|pkwy|circle|cir|boulevard|blvd|lane|ln|way)\b[,.\s]+[\w\s]+[,.\s]+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b',
        re.IGNORECASE
    )

    # Simplified address pattern — number + street name + street type (no zip required)
    ADDRESS_SIMPLE_PATTERN = re.compile(
        r'\b\d{1,5}\s+[A-Za-z][A-Za-z\s]{1,30}(?:street|st|avenue|ave|road|rd|highway|hwy|drive|dr|court|ct|boulevard|blvd|lane|ln|way|place|pl|circle|cir|parkway|pkwy)\b',
        re.IGNORECASE
    )

    # Person name patterns — ordered from high to low confidence
    # 1. Salutation-anchored: "Dear John", "Mr. Smith", "Dr. Gulgowski"
    PERSON_NAME_SALUTATION = re.compile(
        r'\b(?:Dear|Hi|Hello|Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?)\s+'
        r'([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2})\b'
    )
    # 2. Self-introduction: "my name is John Smith" or "I'm Garrick Murray"
    PERSON_NAME_INTRO = re.compile(
        r'\bmy\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        r'|\bI\'?m\s+([A-Z][a-z]{2,20}(?:\s+[A-Z][a-z]{2,20})+)\b'
    )
    # 3. Role-context: "for patient Leuschke", "client Garrick Murray"
    PERSON_NAME_CONTEXT = re.compile(
        r'\b(?:patient|client|student|user|member|subscriber|employee|recipient|'
        r'caller|sender|buyer|seller|applicant|owner|holder|resident|tenant)\s+'
        r'([A-Z][a-z]{2,20}(?:\s+[A-Z][a-z]{2,20}){0,2})\b'
    )
    # Legacy alias kept for backward compatibility
    PERSON_NAME_PATTERN = PERSON_NAME_INTRO

    # GPS coordinates: [-71.6702,-107.6572] bracket format
    GPS_PATTERN = re.compile(
        r'\[[-]?\d{1,3}\.\d{1,8},\s*[-]?\d{1,3}\.\d{1,8}\]'
    )

    # IMEI: dashed format XX-XXXXXX-XXXXXX-X or plain 15 digits
    IMEI_PATTERN = re.compile(
        r'\b\d{2}-\d{6}-\d{6}-\d\b'   # dashed: 06-184755-866851-3
        r'|\b\d{15}\b'                  # plain 15 digits (490154203237518)
    )

    # Ethereum wallet address: 0x + 40 hex chars
    ETHEREUM_PATTERN = re.compile(
        r'\b0x[0-9A-Fa-f]{40}\b'
    )

    # Bitcoin address: P2PKH (1...), P2SH (3...), or bech32 (bc1...)
    BITCOIN_PATTERN = re.compile(
        r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'
        r'|\bbc1[a-z0-9]{39,59}\b'
    )

    # Password-in-context: "password: yZqd7gHyZq91", "Your token: h4yW2I4iydWh"
    PASSWORD_CONTEXT_PATTERN = re.compile(
        r'(?:password|passcode|passphrase|pin|secret\s+key|api\s+secret|token|credential)[s]?'
        r'[\s:=\-]{1,5}([A-Za-z0-9!@#$%^&*()\[\]_\-+=<>?]{8,80})',
        re.IGNORECASE
    )

    # URL pattern: http/https URLs and www. addresses
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"\']{8,}'                           # http/https URLs
        r'|\bwww\.[a-zA-Z0-9.-]+\.[a-z]{2,}[^\s<>"\']*',   # www. URLs
        re.IGNORECASE
    )

    # Username / handle: @handle or "username: X"
    USERNAME_PATTERN = re.compile(
        r'@[A-Za-z0-9_]{3,30}\b'
        r'|(?:username|user\s*name|login|handle)[\s:=]+["\']?([A-Za-z0-9_.\-]{3,30})["\']?',
        re.IGNORECASE
    )

    # Masked card/account numbers: XXXX-XXXX-XXXX-1234, ****1234, 4111 **** **** 1234
    MASKED_NUMBER_PATTERN = re.compile(
        r'\b(?:X{4}[\s-]){2,3}X{4}[\s-]\d{4}\b'         # XXXX-XXXX-XXXX-1234
        r'|\*{4}[\s-]?\*{4}[\s-]?\*{4}[\s-]?\d{4}\b'    # ****-****-****-1234
        r'|\b\d{4}[\s-]\*{4}[\s-]\*{4}[\s-]\d{4}\b',    # 4111 **** **** 1234
        re.IGNORECASE
    )

    # Browser user agent: Mozilla/5.0 (...)
    USER_AGENT_PATTERN = re.compile(
        r'Mozilla/\d\.\d\s*\([^)]{10,}\)',
        re.IGNORECASE
    )

    # Date of Birth pattern (various formats)
    _MONTHS = r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    DOB_PATTERN = re.compile(
        r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)[0-9]{2}\b|'  # MM/DD/YYYY
        r'\b(?:19|20)[0-9]{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])\b|'  # YYYY/MM/DD
        r'\b(?:0?[1-9]|[12][0-9]|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)[0-9]{2}\b|'  # DD/MM/YYYY
        # Text-month formats: "March 5, 1985" / "5th March 1985" / "5 March 1985"
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?,?\s+(?:19|20)[0-9]{2}\b|'
        r'\b(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\s+'
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\s+(?:19|20)[0-9]{2}\b',
        re.IGNORECASE
    )

    # Medical Record Number pattern (common formats)
    MRN_PATTERN = re.compile(
        r'\b(?:MRN|mrn|Medical Record|Patient ID)[:\s#]*[A-Z0-9]{6,12}\b',
        re.IGNORECASE
    )

    # Health Insurance ID pattern
    HEALTH_INSURANCE_PATTERN = re.compile(
        r'\b(?:Member ID|Policy|Insurance ID|Subscriber)[:\s#]*[A-Z0-9]{8,15}\b',
        re.IGNORECASE
    )

    # MAC Address pattern
    MAC_ADDRESS_PATTERN = re.compile(
        r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b|'  # XX:XX:XX:XX:XX:XX
        r'\b(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}\b'  # XXXX.XXXX.XXXX
    )

    # VIN (Vehicle Identification Number) pattern - 17 characters
    VIN_PATTERN = re.compile(
        r'\b[A-HJ-NPR-Z0-9]{17}\b'  # 17 chars, excluding I, O, Q
    )

    # License Plate pattern (US - various state formats)
    LICENSE_PLATE_PATTERN = re.compile(
        r'\b[A-Z0-9]{1,4}[\s-]?[A-Z0-9]{2,4}[\s-]?[A-Z0-9]{0,4}\b'
    )

    # ------------------------------------------------------------------
    # Regional identifier patterns
    # ------------------------------------------------------------------
    # Tuple layout: (config_key, compiled_regex, severity, label, validator_fn_or_None)
    # Validator runs on the matched string and returns True if the match is valid.
    # All regional types default to disabled; they are explicitly enabled per deployment.
    # ------------------------------------------------------------------

    REGIONAL_PATTERNS: List[Tuple[str, "re.Pattern", Severity, str, Optional[Callable[[str], bool]]]] = [
        # ---------- India ----------
        # PAN: 5 letters + 4 digits + 1 letter (AAAPL1234C)
        ('in_pan',
         re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'),
         Severity.HIGH, 'IN_PAN', None),
        # Aadhaar: 12 digits (grouped 4-4-4), Verhoeff checksum
        ('in_aadhaar',
         re.compile(r'\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b'),
         Severity.HIGH, 'IN_AADHAAR',
         lambda s: verhoeff_check(re.sub(r'[\s-]', '', s))),
        # GSTIN: 15-char state-PAN-entity-Z-check
        ('in_gstin',
         re.compile(r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b'),
         Severity.HIGH, 'IN_GSTIN', None),
        # DIN: Director Identification Number - 8 digits
        ('in_din',
         re.compile(r'\b(?:DIN|din)[:\s#]*(\d{8})\b'),
         Severity.MEDIUM, 'IN_DIN', None),
        # IFSC: 4 letters + 0 + 6 alphanumeric
        ('in_ifsc',
         re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b'),
         Severity.MEDIUM, 'IN_IFSC', None),
        # UPI VPA: user@provider. Negative lookahead for `.` rejects
        # matches that are actually the local@domain of a full email
        # (e.g. "alice@company.com" - the first dot shows it's an email).
        ('in_upi',
         re.compile(r'\b[A-Za-z0-9._-]{3,}@[A-Za-z]{2,20}\b(?!\.)'),
         Severity.MEDIUM, 'IN_UPI',
         lambda s: '.' not in s.split('@', 1)[1]),

        # ---------- United Kingdom ----------
        # NINO: 2 letters + 6 digits + [A-D] (letters exclude DFIQUV prefix, O second letter)
        ('uk_nino',
         re.compile(r'\b[ABCEGHJ-NOPRSTW-Z][ABCEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b'),
         Severity.HIGH, 'UK_NINO', None),
        # UTR: "UTR" keyword + 10 digits
        ('uk_utr',
         re.compile(r'\b(?:UTR|utr)[:\s#]*(\d{10})\b'),
         Severity.MEDIUM, 'UK_UTR', None),
        # NHS: 10 digits, mod-11
        ('uk_nhs',
         re.compile(r'\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b'),
         Severity.HIGH, 'UK_NHS',
         lambda s: mod11_nhs_check(re.sub(r'[\s-]', '', s))),
        # UK VAT: GB + 9 digits (or 12 with branch trader)
        ('uk_vat',
         re.compile(r'\bGB\s?\d{9}(?:\s?\d{3})?\b'),
         Severity.MEDIUM, 'UK_VAT', None),

        # ---------- EU (country-specific) ----------
        # Germany VAT
        ('de_vat',
         re.compile(r'\bDE\s?\d{9}\b'),
         Severity.MEDIUM, 'DE_VAT', None),
        # France VAT (2-char prefix can be digits or letters)
        ('fr_vat',
         re.compile(r'\bFR\s?[A-HJ-NP-Z0-9]{2}\s?\d{9}\b'),
         Severity.MEDIUM, 'FR_VAT', None),
        # Italy VAT (11 digits)
        ('it_vat',
         re.compile(r'\bIT\s?\d{11}\b'),
         Severity.MEDIUM, 'IT_VAT', None),
        # Netherlands VAT (9 digits + B + 2 digits)
        ('nl_vat',
         re.compile(r'\bNL\s?\d{9}B\d{2}\b'),
         Severity.MEDIUM, 'NL_VAT', None),
        # Spain VAT - first and last positions can be any letter (CIF/NIF/NIE) or digit
        ('es_vat',
         re.compile(r'\bES\s?[A-Z0-9]\d{7}[A-Z0-9]\b'),
         Severity.MEDIUM, 'ES_VAT', None),
        # Belgium VAT (BE + 10 digits)
        ('be_vat',
         re.compile(r'\bBE\s?0?\d{9}\b'),
         Severity.MEDIUM, 'BE_VAT', None),
        # Netherlands BSN (9 digits, 11-test)
        ('nl_bsn',
         re.compile(r'\b[1-9]\d{8}\b'),
         Severity.HIGH, 'NL_BSN',
         lambda s: _bsn_11_check(s)),
        # Italy Codice Fiscale (6 letters + 2 digits + 1 letter + 2 digits + 1 letter + 3 alnum + 1 letter)
        ('it_cf',
         re.compile(r'\b[A-Z]{6}\d{2}[A-EHLMPRST]\d{2}[A-Z]\d{3}[A-Z]\b'),
         Severity.HIGH, 'IT_CF', None),
        # Denmark CPR (DDMMYY-XXXX)
        ('dk_cpr',
         re.compile(r'\b(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{2}[-\s]?\d{4}\b'),
         Severity.HIGH, 'DK_CPR', None),
        # Romania CNP (1-8 + YYMMDD + 2-digit county + 3 serial + check)
        ('ro_cnp',
         re.compile(r'\b[1-8]\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{6}\b'),
         Severity.HIGH, 'RO_CNP', None),
        # Poland PESEL (11 digits; date-prefixed)
        ('pl_pesel',
         re.compile(r'\b\d{2}(?:[02][1-9]|[13][0-2])(?:0[1-9]|[12]\d|3[01])\d{5}\b'),
         Severity.HIGH, 'PL_PESEL', None),
        # Greece AMKA (11 digits starting DDMMYY)
        ('gr_amka',
         re.compile(r'\b(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{2}\d{5}\b'),
         Severity.HIGH, 'GR_AMKA', None),

        # ---------- APAC ----------
        # Singapore NRIC / FIN
        ('sg_nric',
         re.compile(r'\b[STFG]\d{7}[A-Z]\b'),
         Severity.HIGH, 'SG_NRIC', None),
        # Malaysia IC (12 digits, grouped YYMMDD-PB-XXXG) - requires the
        # first separator so bare 12-digit sequences don't collide.
        ('my_ic',
         re.compile(r'\b\d{6}[-\s]\d{2}[-\s]?\d{4}\b'),
         Severity.HIGH, 'MY_IC', None),
        # Thailand National ID (13 digits) - requires the first separator so
        # unrelated 13-digit sequences (CNP, etc.) don't collide.
        ('th_id',
         re.compile(r'\b\d{1}[-\s]\d{4}[-\s]?\d{5}[-\s]?\d{2}[-\s]?\d{1}\b'),
         Severity.HIGH, 'TH_ID', None),
        # Philippines TIN (9 + 3 digit branch) - requires the first separator
        # so 12-digit numeric-only sequences don't collide.
        ('ph_tin',
         re.compile(r'\b\d{3}[-\s]\d{3}[-\s]?\d{3}[-\s]?\d{3}\b'),
         Severity.HIGH, 'PH_TIN', None),
        # Indonesia KTP / NIK (16 digits)
        ('id_ktp',
         re.compile(r'\b\d{16}\b'),
         Severity.HIGH, 'ID_KTP',
         lambda s: _ktp_date_sanity(s)),
        # Taiwan ID (1 letter + [12] + 8 digits)
        ('tw_id',
         re.compile(r'\b[A-Z][12]\d{8}\b'),
         Severity.HIGH, 'TW_ID', None),
        # Japan MyNumber (12 digits)
        ('jp_mynumber',
         re.compile(r'\b(?:MyNumber|マイナンバー|個人番号)[:\s#]*(\d{4}[\s-]?\d{4}[\s-]?\d{4})\b'),
         Severity.HIGH, 'JP_MYNUMBER', None),
        # Australia TFN (8-9 digits, weighted mod-11) - REQUIRES the "TFN" keyword
        # anchor to prevent false positives on unrelated numeric sequences that
        # happen to pass the mod-11 check.
        ('au_tfn',
         re.compile(r'\bTFN[:\s#]*(\d{3}[\s-]?\d{3}[\s-]?\d{2,3})\b', re.IGNORECASE),
         Severity.HIGH, 'AU_TFN',
         lambda s: _au_tfn_check(s)),
        # Australia ABN (11 digits, weighted mod-89) - REQUIRES the "ABN" keyword
        ('au_abn',
         re.compile(r'\bABN[:\s#]*(\d{2}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3})\b', re.IGNORECASE),
         Severity.MEDIUM, 'AU_ABN',
         lambda s: _au_abn_check(s)),
        # Australia ACN (9 digits, weighted mod-10 check digit) - REQUIRES the
        # "ACN" keyword; validator now enforces the check-digit algorithm.
        ('au_acn',
         re.compile(r'\bACN[:\s#]*(\d{3}[\s-]?\d{3}[\s-]?\d{3})\b', re.IGNORECASE),
         Severity.MEDIUM, 'AU_ACN',
         lambda s: _au_acn_check(s)),
        # Hong Kong HKID (1-2 letters + 6 digits + check in brackets)
        ('hk_id',
         re.compile(r'\b[A-Z]{1,2}\d{6}[\(\[]([A0-9])[\)\]]'),
         Severity.HIGH, 'HK_ID', None),

        # ---------- Americas ----------
        # Canada SIN (9 digits, Luhn) - REQUIRES a "SIN" or "NAS" keyword
        # anchor to prevent false positives on unrelated 9-digit sequences
        # (product IDs, batch numbers) that happen to pass the Luhn check.
        ('ca_sin',
         re.compile(r'\b(?:SIN|NAS)[:\s#]*(\d{3}[-\s]?\d{3}[-\s]?\d{3})\b', re.IGNORECASE),
         Severity.HIGH, 'CA_SIN',
         lambda s: mod10_sin_check(s)),
        # Brazil CPF (11 digits, 2 check digits)
        ('br_cpf',
         re.compile(r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b'),
         Severity.HIGH, 'BR_CPF',
         lambda s: cpf_check(s)),
        # Brazil CNPJ (14 digits, 2 check digits)
        ('br_cnpj',
         re.compile(r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b'),
         Severity.HIGH, 'BR_CNPJ',
         lambda s: cnpj_check(s)),
        # Mexico CURP (18-char identifier)
        ('mx_curp',
         re.compile(r'\b[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d\b'),
         Severity.HIGH, 'MX_CURP', None),
        # Mexico RFC (13 chars for individuals, 12 for companies)
        ('mx_rfc',
         re.compile(r'\b[A-Z&Ñ]{3,4}\d{6}[A-Z0-9]{3}\b'),
         Severity.HIGH, 'MX_RFC', None),
        # Spain NIF/NIE (8 digits + letter, or X/Y/Z + 7 + letter)
        ('es_nif',
         re.compile(r'\b[XYZ]?\d{7,8}[A-Z]\b'),
         Severity.HIGH, 'ES_NIF',
         lambda s: nif_check(s)),
    ]

    # Default type configuration
    DEFAULT_TYPE_CONFIG = {
        # Identity Documents
        'ssn': {'enabled': True, 'action': 'REDACT'},
        'passport': {'enabled': False, 'action': 'REDACT'},
        'drivers_license': {'enabled': False, 'action': 'REDACT'},
        'national_id': {'enabled': False, 'action': 'REDACT'},
        'tax_id': {'enabled': False, 'action': 'REDACT'},
        # Financial
        'credit_card': {'enabled': True, 'action': 'REDACT'},
        'bank_account': {'enabled': True, 'action': 'REDACT'},
        'iban': {'enabled': True, 'action': 'REDACT'},
        # Contact
        'email': {'enabled': True, 'action': 'REDACT'},
        'phone': {'enabled': True, 'action': 'REDACT'},
        'address': {'enabled': False, 'action': 'REDACT'},
        # Personal
        'date_of_birth': {'enabled': True, 'action': 'REDACT'},
        'person_name': {'enabled': True, 'action': 'REDACT'},
        # Healthcare (PHI)
        'medical_record': {'enabled': False, 'action': 'REDACT'},
        'health_insurance_id': {'enabled': False, 'action': 'REDACT'},
        # Technical
        'ip_address': {'enabled': True, 'action': 'REDACT'},
        'mac_address': {'enabled': True, 'action': 'REDACT'},
        'api_key': {'enabled': True, 'action': 'REDACT'},
        # Vehicle
        'vin': {'enabled': False, 'action': 'REDACT'},
        'license_plate': {'enabled': False, 'action': 'REDACT'},
        # Crypto / device identifiers
        'gps_coordinates': {'enabled': True, 'action': 'REDACT'},
        'imei': {'enabled': True, 'action': 'REDACT'},
        'ethereum_address': {'enabled': True, 'action': 'REDACT'},
        'bitcoin_address': {'enabled': True, 'action': 'REDACT'},
        'password': {'enabled': True, 'action': 'REDACT'},
        # Web / account identifiers
        'url': {'enabled': True, 'action': 'REDACT'},
        'username': {'enabled': True, 'action': 'REDACT'},
        'masked_number': {'enabled': True, 'action': 'REDACT'},
        'user_agent': {'enabled': True, 'action': 'REDACT'},
        # Regional identifiers - default disabled; enable per deployment
        # India
        'in_pan': {'enabled': False, 'action': 'REDACT'},
        'in_aadhaar': {'enabled': False, 'action': 'REDACT'},
        'in_gstin': {'enabled': False, 'action': 'REDACT'},
        'in_din': {'enabled': False, 'action': 'REDACT'},
        'in_ifsc': {'enabled': False, 'action': 'REDACT'},
        'in_upi': {'enabled': False, 'action': 'REDACT'},
        # United Kingdom
        'uk_nino': {'enabled': False, 'action': 'REDACT'},
        'uk_utr': {'enabled': False, 'action': 'REDACT'},
        'uk_nhs': {'enabled': False, 'action': 'REDACT'},
        'uk_vat': {'enabled': False, 'action': 'REDACT'},
        # EU
        'de_vat': {'enabled': False, 'action': 'REDACT'},
        'fr_vat': {'enabled': False, 'action': 'REDACT'},
        'it_vat': {'enabled': False, 'action': 'REDACT'},
        'nl_vat': {'enabled': False, 'action': 'REDACT'},
        'es_vat': {'enabled': False, 'action': 'REDACT'},
        'be_vat': {'enabled': False, 'action': 'REDACT'},
        'nl_bsn': {'enabled': False, 'action': 'REDACT'},
        'it_cf': {'enabled': False, 'action': 'REDACT'},
        'dk_cpr': {'enabled': False, 'action': 'REDACT'},
        'ro_cnp': {'enabled': False, 'action': 'REDACT'},
        'pl_pesel': {'enabled': False, 'action': 'REDACT'},
        'gr_amka': {'enabled': False, 'action': 'REDACT'},
        'es_nif': {'enabled': False, 'action': 'REDACT'},
        # APAC
        'sg_nric': {'enabled': False, 'action': 'REDACT'},
        'my_ic': {'enabled': False, 'action': 'REDACT'},
        'th_id': {'enabled': False, 'action': 'REDACT'},
        'ph_tin': {'enabled': False, 'action': 'REDACT'},
        'id_ktp': {'enabled': False, 'action': 'REDACT'},
        'tw_id': {'enabled': False, 'action': 'REDACT'},
        'jp_mynumber': {'enabled': False, 'action': 'REDACT'},
        'au_tfn': {'enabled': False, 'action': 'REDACT'},
        'au_abn': {'enabled': False, 'action': 'REDACT'},
        'au_acn': {'enabled': False, 'action': 'REDACT'},
        'hk_id': {'enabled': False, 'action': 'REDACT'},
        # Americas
        'ca_sin': {'enabled': False, 'action': 'REDACT'},
        'br_cpf': {'enabled': False, 'action': 'REDACT'},
        'br_cnpj': {'enabled': False, 'action': 'REDACT'},
        'mx_curp': {'enabled': False, 'action': 'REDACT'},
        'mx_rfc': {'enabled': False, 'action': 'REDACT'},
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PII detector with configuration.

        Args:
            config: Configuration dict with keys:
                - action: "REDACT" or "BLOCK" (default: "REDACT") - default action
                - enabled: bool (default: True)
                - redaction_strategy: "full" or "partial" (default: "full")
                - skip_private_ips: bool (default: True)
                - types: dict of per-type configurations (optional)
                    Each type can have:
                    - enabled: bool
                    - action: "REDACT" or "BLOCK"
        """
        self.config = config
        self.default_action = config.get('action', 'REDACT')
        self.enabled = config.get('enabled', True)
        self.redaction_strategy = config.get('redaction_strategy', 'full')
        self.skip_private_ips = config.get('skip_private_ips', True)
        # When luhn_strict=False, skip Luhn validation for synthetic/test card numbers
        self.luhn_strict = config.get('luhn_strict', True)

        # Parse per-type configurations
        types_config = config.get('types', {})
        self.type_configs = {}
        for pii_type, default_cfg in self.DEFAULT_TYPE_CONFIG.items():
            if pii_type in types_config:
                type_cfg = types_config[pii_type]
                if isinstance(type_cfg, bool):
                    type_cfg = {'enabled': type_cfg}
                self.type_configs[pii_type] = {
                    'enabled': type_cfg.get('enabled', default_cfg['enabled']),
                    'action': type_cfg.get('action', self.default_action),
                }
            else:
                # Use defaults with the configured default action
                self.type_configs[pii_type] = {
                    'enabled': default_cfg['enabled'],
                    'action': self.default_action,
                }

        # Load custom regex patterns from multilingual packs (config key: "patterns")
        custom_patterns = config.get('patterns', {})
        self.custom_compiled_patterns: List[Tuple[str, 're.Pattern']] = []
        for name, regex in custom_patterns.items():
            if isinstance(regex, str):
                try:
                    self.custom_compiled_patterns.append((name, re.compile(regex, re.IGNORECASE)))
                except re.error:
                    pass  # skip invalid regex

    def _is_type_enabled(self, pii_type: str) -> bool:
        """Check if a PII type is enabled for detection."""
        return self.type_configs.get(pii_type, {}).get('enabled', True)

    def _get_type_action(self, pii_type: str) -> str:
        """Get the action for a specific PII type."""
        return self.type_configs.get(pii_type, {}).get('action', self.default_action)

    @staticmethod
    def is_private_ip(ip: str) -> bool:
        """
        Check if IPv4 address is private/internal.

        Returns True for:
        - 10.0.0.0/8
        - 172.16.0.0/12
        - 192.168.0.0/16
        - 127.0.0.0/8 (localhost)
        - 169.254.0.0/16 (link-local)
        """
        try:
            parts = [int(p) for p in ip.split('.')]
            if len(parts) != 4:
                return False

            # 10.0.0.0/8
            if parts[0] == 10:
                return True
            # 172.16.0.0/12
            if parts[0] == 172 and 16 <= parts[1] <= 31:
                return True
            # 192.168.0.0/16
            if parts[0] == 192 and parts[1] == 168:
                return True
            # 127.0.0.0/8 (localhost)
            if parts[0] == 127:
                return True
            # 169.254.0.0/16 (link-local)
            if parts[0] == 169 and parts[1] == 254:
                return True

            return False
        except (ValueError, AttributeError):
            return False

    def detect(self, text: str) -> DetectorResult:
        """
        Detect PII in text.

        Args:
            text: Input text to scan

        Returns:
            DetectorResult with findings
        """
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        # (start, end, type, matched_text, config_type) - config_type maps to type_configs key
        pii_spans: List[Tuple[int, int, str, str, str]] = []
        blocked_types: List[str] = []  # Track which types triggered BLOCK

        # Detect emails (if enabled)
        if self._is_type_enabled('email'):
            for match in self.EMAIL_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.email",
                    severity=Severity.MEDIUM,
                    message="Email address detected"
                ))
                pii_spans.append((match.start(), match.end(), "EMAIL", match.group(), 'email'))
                if self._get_type_action('email') == 'BLOCK':
                    blocked_types.append('email')

        # Detect phone numbers (if enabled)
        if self._is_type_enabled('phone'):
            for match in self.PHONE_PATTERN.finditer(text):
                # Context check: skip if preceded by non-phone keywords
                prefix = text[max(0, match.start() - 30):match.start()]
                if self._PHONE_FALSE_POSITIVE_CONTEXT.search(prefix):
                    continue

                rule_hits.append(RuleHit(
                    rule_id="pii.phone",
                    severity=Severity.MEDIUM,
                    message="Phone number detected"
                ))
                pii_spans.append((match.start(), match.end(), "PHONE", match.group(), 'phone'))
                if self._get_type_action('phone') == 'BLOCK':
                    blocked_types.append('phone')

        # Detect IPv4 addresses (if enabled)
        if self._is_type_enabled('ip_address'):
            for match in self.IPV4_PATTERN.finditer(text):
                # Validate that it's a reasonable IP (not like 999.999.999.999)
                ip_str = match.group()
                try:
                    ip_parts = ip_str.split('.')
                    if all(0 <= int(part) <= 255 for part in ip_parts):
                        # Skip private IPs if configured
                        if self.skip_private_ips and self.is_private_ip(ip_str):
                            continue

                        rule_hits.append(RuleHit(
                            rule_id="pii.ipv4",
                            severity=Severity.LOW,
                            message="IPv4 address detected"
                        ))
                        pii_spans.append((match.start(), match.end(), "IPV4", ip_str, 'ip_address'))
                        if self._get_type_action('ip_address') == 'BLOCK':
                            blocked_types.append('ip_address')
                except (ValueError, AttributeError):
                    continue

            # Detect IPv6 addresses
            for match in self.IPV6_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.ipv6",
                    severity=Severity.LOW,
                    message="IPv6 address detected"
                ))
                pii_spans.append((match.start(), match.end(), "IPV6", match.group(), 'ip_address'))
                if self._get_type_action('ip_address') == 'BLOCK':
                    blocked_types.append('ip_address')

        # Detect SSN (Social Security Numbers) (if enabled)
        if self._is_type_enabled('ssn'):
            for match in self.SSN_PATTERN.finditer(text):
                # Context check: skip if preceded by date/non-SSN keywords
                prefix = text[max(0, match.start() - 30):match.start()]
                if self._SSN_FALSE_POSITIVE_CONTEXT.search(prefix):
                    continue

                # Skip if the matched value looks like a date (MM-DD-YYYY)
                area = int(match.group(1))
                group = int(match.group(3))
                if 1 <= area <= 12 and 1 <= group <= 31:
                    # Could be a date -- check if suffix looks like a year
                    serial = match.group(4)
                    if serial.startswith('19') or serial.startswith('20'):
                        continue

                rule_hits.append(RuleHit(
                    rule_id="pii.ssn",
                    severity=Severity.HIGH,
                    message="Social Security Number detected"
                ))
                pii_spans.append((match.start(), match.end(), "SSN", match.group(), 'ssn'))
                if self._get_type_action('ssn') == 'BLOCK':
                    blocked_types.append('ssn')

        # Detect API keys (if enabled)
        if self._is_type_enabled('api_key'):
            for pattern, key_type, severity in self.API_KEY_PATTERNS:
                for match in pattern.finditer(text):
                    rule_hits.append(RuleHit(
                        rule_id=f"pii.api_key.{key_type}",
                        severity=severity,
                        message=f"API key detected ({key_type})"
                    ))
                    pii_spans.append((match.start(), match.end(), "API_KEY", match.group(), 'api_key'))
                    if self._get_type_action('api_key') == 'BLOCK':
                        blocked_types.append('api_key')

        # Detect credit cards (if enabled)
        if self._is_type_enabled('credit_card'):
            for match in self.CARD_PATTERN.finditer(text):
                # Extract just the digits
                card_number = re.sub(r'[^0-9]', '', match.group())

                # Validate with Luhn check (skip when luhn_strict=False for synthetic data)
                if 13 <= len(card_number) <= 19 and (not self.luhn_strict or luhn_check(card_number)):
                    rule_hits.append(RuleHit(
                        rule_id="pii.credit_card",
                        severity=Severity.HIGH,
                        message="Credit card number detected"
                    ))
                    pii_spans.append((match.start(), match.end(), "CREDIT_CARD", match.group(), 'credit_card'))
                    if self._get_type_action('credit_card') == 'BLOCK':
                        blocked_types.append('credit_card')

        # Detect passport numbers (if enabled)
        if self._is_type_enabled('passport'):
            for match in self.PASSPORT_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.passport",
                    severity=Severity.HIGH,
                    message="Passport number detected"
                ))
                pii_spans.append((match.start(), match.end(), "PASSPORT", match.group(), 'passport'))
                if self._get_type_action('passport') == 'BLOCK':
                    blocked_types.append('passport')

        # Detect driver's license (if enabled)
        if self._is_type_enabled('drivers_license'):
            for match in self.DRIVERS_LICENSE_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.drivers_license",
                    severity=Severity.HIGH,
                    message="Driver's license number detected"
                ))
                pii_spans.append((match.start(), match.end(), "DRIVERS_LICENSE", match.group(), 'drivers_license'))
                if self._get_type_action('drivers_license') == 'BLOCK':
                    blocked_types.append('drivers_license')

        # Detect national ID (if enabled)
        if self._is_type_enabled('national_id'):
            for match in self.NATIONAL_ID_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.national_id",
                    severity=Severity.HIGH,
                    message="National ID number detected"
                ))
                pii_spans.append((match.start(), match.end(), "NATIONAL_ID", match.group(), 'national_id'))
                if self._get_type_action('national_id') == 'BLOCK':
                    blocked_types.append('national_id')

        # Detect tax ID (ITIN/EIN) (if enabled)
        if self._is_type_enabled('tax_id'):
            for match in self.TAX_ID_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.tax_id",
                    severity=Severity.HIGH,
                    message="Tax ID (ITIN/EIN) detected"
                ))
                pii_spans.append((match.start(), match.end(), "TAX_ID", match.group(), 'tax_id'))
                if self._get_type_action('tax_id') == 'BLOCK':
                    blocked_types.append('tax_id')

        # Detect bank account numbers (if enabled)
        if self._is_type_enabled('bank_account'):
            for match in self.BANK_ACCOUNT_PATTERN.finditer(text):
                # Basic validation - must be 8-17 digits
                account = match.group()
                if 8 <= len(account) <= 17:
                    rule_hits.append(RuleHit(
                        rule_id="pii.bank_account",
                        severity=Severity.HIGH,
                        message="Bank account number detected"
                    ))
                    pii_spans.append((match.start(), match.end(), "BANK_ACCOUNT", account, 'bank_account'))
                    if self._get_type_action('bank_account') == 'BLOCK':
                        blocked_types.append('bank_account')

        # Detect IBAN (if enabled)
        if self._is_type_enabled('iban'):
            for match in self.IBAN_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.iban",
                    severity=Severity.HIGH,
                    message="IBAN detected"
                ))
                pii_spans.append((match.start(), match.end(), "IBAN", match.group(), 'iban'))
                if self._get_type_action('iban') == 'BLOCK':
                    blocked_types.append('iban')

        # Detect physical addresses (if enabled)
        if self._is_type_enabled('address'):
            matched_address_spans: set = set()
            for match in self.ADDRESS_PATTERN.finditer(text):
                matched_address_spans.add((match.start(), match.end()))
                rule_hits.append(RuleHit(
                    rule_id="pii.address",
                    severity=Severity.MEDIUM,
                    message="Physical address detected"
                ))
                pii_spans.append((match.start(), match.end(), "ADDRESS", match.group(), 'address'))
                if self._get_type_action('address') == 'BLOCK':
                    blocked_types.append('address')
            # Also match partial addresses (no zip required)
            for match in self.ADDRESS_SIMPLE_PATTERN.finditer(text):
                overlap = any(s <= match.start() < e or s < match.end() <= e
                              for s, e in matched_address_spans)
                if not overlap:
                    rule_hits.append(RuleHit(
                        rule_id="pii.address",
                        severity=Severity.MEDIUM,
                        message="Physical address detected"
                    ))
                    pii_spans.append((match.start(), match.end(), "ADDRESS", match.group(), 'address'))
                    if self._get_type_action('address') == 'BLOCK':
                        blocked_types.append('address')

        # Detect person names (if enabled) — three complementary patterns
        if self._is_type_enabled('person_name'):
            _name_spans: set = set()
            for _pattern in (
                self.PERSON_NAME_SALUTATION,
                self.PERSON_NAME_INTRO,
                self.PERSON_NAME_CONTEXT,
            ):
                for match in _pattern.finditer(text):
                    span = (match.start(), match.end())
                    if span in _name_spans:
                        continue
                    _name_spans.add(span)
                    rule_hits.append(RuleHit(
                        rule_id="pii.person_name",
                        severity=Severity.MEDIUM,
                        message="Person name detected"
                    ))
                    pii_spans.append((match.start(), match.end(), "PERSON_NAME", match.group(), 'person_name'))
                    if self._get_type_action('person_name') == 'BLOCK':
                        blocked_types.append('person_name')

        # Detect date of birth (if enabled)
        if self._is_type_enabled('date_of_birth'):
            for match in self.DOB_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.date_of_birth",
                    severity=Severity.MEDIUM,
                    message="Date of birth detected"
                ))
                pii_spans.append((match.start(), match.end(), "DOB", match.group(), 'date_of_birth'))
                if self._get_type_action('date_of_birth') == 'BLOCK':
                    blocked_types.append('date_of_birth')

        # Detect medical record numbers (if enabled)
        if self._is_type_enabled('medical_record'):
            for match in self.MRN_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.medical_record",
                    severity=Severity.HIGH,
                    message="Medical record number detected"
                ))
                pii_spans.append((match.start(), match.end(), "MRN", match.group(), 'medical_record'))
                if self._get_type_action('medical_record') == 'BLOCK':
                    blocked_types.append('medical_record')

        # Detect health insurance ID (if enabled)
        if self._is_type_enabled('health_insurance_id'):
            for match in self.HEALTH_INSURANCE_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.health_insurance_id",
                    severity=Severity.HIGH,
                    message="Health insurance ID detected"
                ))
                pii_spans.append((match.start(), match.end(), "HEALTH_INSURANCE", match.group(), 'health_insurance_id'))
                if self._get_type_action('health_insurance_id') == 'BLOCK':
                    blocked_types.append('health_insurance_id')

        # Detect MAC addresses (if enabled)
        if self._is_type_enabled('mac_address'):
            for match in self.MAC_ADDRESS_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.mac_address",
                    severity=Severity.LOW,
                    message="MAC address detected"
                ))
                pii_spans.append((match.start(), match.end(), "MAC_ADDRESS", match.group(), 'mac_address'))
                if self._get_type_action('mac_address') == 'BLOCK':
                    blocked_types.append('mac_address')

        # Detect VIN (if enabled)
        if self._is_type_enabled('vin'):
            for match in self.VIN_PATTERN.finditer(text):
                vin = match.group()
                # Additional VIN validation - check length and excluded characters
                if len(vin) == 17:
                    rule_hits.append(RuleHit(
                        rule_id="pii.vin",
                        severity=Severity.MEDIUM,
                        message="Vehicle Identification Number detected"
                    ))
                    pii_spans.append((match.start(), match.end(), "VIN", vin, 'vin'))
                    if self._get_type_action('vin') == 'BLOCK':
                        blocked_types.append('vin')

        # Detect license plates (if enabled)
        if self._is_type_enabled('license_plate'):
            for match in self.LICENSE_PLATE_PATTERN.finditer(text):
                plate = match.group()
                # Basic validation - 2-8 characters
                if 2 <= len(plate.replace(' ', '').replace('-', '')) <= 8:
                    rule_hits.append(RuleHit(
                        rule_id="pii.license_plate",
                        severity=Severity.MEDIUM,
                        message="License plate detected"
                    ))
                    pii_spans.append((match.start(), match.end(), "LICENSE_PLATE", plate, 'license_plate'))
                    if self._get_type_action('license_plate') == 'BLOCK':
                        blocked_types.append('license_plate')

        # Detect GPS coordinates (if enabled)
        if self._is_type_enabled('gps_coordinates'):
            for match in self.GPS_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.gps_coordinates",
                    severity=Severity.MEDIUM,
                    message="GPS coordinates detected"
                ))
                pii_spans.append((match.start(), match.end(), "GPS", match.group(), 'gps_coordinates'))
                if self._get_type_action('gps_coordinates') == 'BLOCK':
                    blocked_types.append('gps_coordinates')

        # Detect IMEI numbers (if enabled)
        if self._is_type_enabled('imei'):
            for match in self.IMEI_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.imei",
                    severity=Severity.MEDIUM,
                    message="IMEI number detected"
                ))
                pii_spans.append((match.start(), match.end(), "IMEI", match.group(), 'imei'))
                if self._get_type_action('imei') == 'BLOCK':
                    blocked_types.append('imei')

        # Detect Ethereum wallet addresses (if enabled)
        if self._is_type_enabled('ethereum_address'):
            for match in self.ETHEREUM_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.ethereum_address",
                    severity=Severity.HIGH,
                    message="Ethereum wallet address detected"
                ))
                pii_spans.append((match.start(), match.end(), "ETHEREUM", match.group(), 'ethereum_address'))
                if self._get_type_action('ethereum_address') == 'BLOCK':
                    blocked_types.append('ethereum_address')

        # Detect Bitcoin addresses (if enabled)
        if self._is_type_enabled('bitcoin_address'):
            for match in self.BITCOIN_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.bitcoin_address",
                    severity=Severity.HIGH,
                    message="Bitcoin address detected"
                ))
                pii_spans.append((match.start(), match.end(), "BITCOIN", match.group(), 'bitcoin_address'))
                if self._get_type_action('bitcoin_address') == 'BLOCK':
                    blocked_types.append('bitcoin_address')

        # Detect passwords in context (if enabled)
        if self._is_type_enabled('password'):
            for match in self.PASSWORD_CONTEXT_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.password",
                    severity=Severity.HIGH,
                    message="Password/credential detected"
                ))
                pii_spans.append((match.start(), match.end(), "PASSWORD", match.group(), 'password'))
                if self._get_type_action('password') == 'BLOCK':
                    blocked_types.append('password')

        # Detect URLs (if enabled)
        if self._is_type_enabled('url'):
            for match in self.URL_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.url",
                    severity=Severity.LOW,
                    message="URL detected"
                ))
                pii_spans.append((match.start(), match.end(), "URL", match.group(), 'url'))
                if self._get_type_action('url') == 'BLOCK':
                    blocked_types.append('url')

        # Detect usernames / handles (if enabled)
        if self._is_type_enabled('username'):
            for match in self.USERNAME_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.username",
                    severity=Severity.MEDIUM,
                    message="Username/handle detected"
                ))
                pii_spans.append((match.start(), match.end(), "USERNAME", match.group(), 'username'))
                if self._get_type_action('username') == 'BLOCK':
                    blocked_types.append('username')

        # Detect masked numbers (if enabled)
        if self._is_type_enabled('masked_number'):
            for match in self.MASKED_NUMBER_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.masked_number",
                    severity=Severity.MEDIUM,
                    message="Masked card/account number detected"
                ))
                pii_spans.append((match.start(), match.end(), "MASKED_NUMBER", match.group(), 'masked_number'))
                if self._get_type_action('masked_number') == 'BLOCK':
                    blocked_types.append('masked_number')

        # Detect browser user agents (if enabled)
        if self._is_type_enabled('user_agent'):
            for match in self.USER_AGENT_PATTERN.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id="pii.user_agent",
                    severity=Severity.LOW,
                    message="Browser user agent detected"
                ))
                pii_spans.append((match.start(), match.end(), "USER_AGENT", match.group(), 'user_agent'))
                if self._get_type_action('user_agent') == 'BLOCK':
                    blocked_types.append('user_agent')

        # Detect regional identifiers (data-driven; each defaults to disabled)
        for config_key, pattern, severity, label, validator in self.REGIONAL_PATTERNS:
            if not self._is_type_enabled(config_key):
                continue
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                # Captured group takes precedence if present (keyword-prefixed patterns)
                try:
                    if match.groups() and match.group(1):
                        matched_text = match.group(1)
                except (IndexError, AttributeError):
                    pass
                if validator is not None and not validator(matched_text):
                    continue
                rule_hits.append(RuleHit(
                    rule_id=f"pii.{config_key}",
                    severity=severity,
                    message=f"Regional identifier detected ({label})"
                ))
                pii_spans.append((match.start(), match.end(), label, match.group(0), config_key))
                if self._get_type_action(config_key) == 'BLOCK':
                    blocked_types.append(config_key)

        # Detect custom multilingual PII patterns
        for name, pattern in self.custom_compiled_patterns:
            for match in pattern.finditer(text):
                rule_hits.append(RuleHit(
                    rule_id=f"pii.custom.{name}",
                    severity=Severity.HIGH,
                    message=f"Custom PII pattern matched ({name})"
                ))
                pii_spans.append((match.start(), match.end(), name, match.group(0), name))
                if self.default_action == 'BLOCK':
                    blocked_types.append(name)

        # No PII found
        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Calculate risk score based on severity
        risk_score = self._calculate_risk_score(rule_hits)

        # Determine decision - if ANY type is configured to BLOCK and was detected, BLOCK
        if blocked_types:
            return DetectorResult(
                decision=Decision.BLOCK,
                risk_score=risk_score,
                rule_hits=rule_hits,
                user_message="Your message contains sensitive information and cannot be processed.",
                developer_message=f"PII blocked: {len(set(blocked_types))} type(s) - {', '.join(set(blocked_types))}"
            )
        else:  # All detected types are configured to REDACT
            sanitized_text = self._redact_pii(text, pii_spans)
            return DetectorResult(
                decision=Decision.REDACT,
                risk_score=risk_score,
                rule_hits=rule_hits,
                sanitized_text=sanitized_text,
                developer_message=f"PII redacted: {len(rule_hits)} item(s)"
            )

    @staticmethod
    def _calculate_risk_score(rule_hits: List[RuleHit]) -> int:
        """Calculate risk score based on rule hits"""
        return calculate_risk_score(rule_hits)

    def _get_redaction_replacement(self, pii_type: str, matched_text: str) -> str:
        """
        Get replacement text based on redaction strategy.

        Args:
            pii_type: Type of PII (EMAIL, PHONE, etc.)
            matched_text: The original matched text

        Returns:
            Redacted replacement string
        """
        if self.redaction_strategy == "partial":
            # Partial redaction - show some characters
            if pii_type == "EMAIL":
                if '@' in matched_text:
                    local, domain = matched_text.split('@', 1)
                    if len(local) > 0:
                        return f"{local[0]}***@{domain}"
                return f"***@{matched_text.split('@')[1]}" if '@' in matched_text else f"[{pii_type}]"

            elif pii_type == "PHONE":
                # Show last 4 digits
                digits = re.sub(r'[^0-9]', '', matched_text)
                if len(digits) >= 4:
                    return f"(***) ***-{digits[-4:]}"
                return f"[{pii_type}]"

            elif pii_type == "CREDIT_CARD":
                # Show last 4 digits
                digits = re.sub(r'[^0-9]', '', matched_text)
                if len(digits) >= 4:
                    return f"****-****-****-{digits[-4:]}"
                return f"[{pii_type}]"

            elif pii_type == "SSN":
                # Show last 4 digits
                digits = re.sub(r'[^0-9]', '', matched_text)
                if len(digits) >= 4:
                    return f"***-**-{digits[-4:]}"
                return f"[{pii_type}]"

            elif pii_type in ["IPV4", "IPV6"]:
                # Mask IP addresses partially
                parts = matched_text.split('.')
                if len(parts) == 4:  # IPv4
                    return f"{parts[0]}.{parts[1]}.x.x"
                return f"[{pii_type}]"

            elif pii_type == "API_KEY":
                # Show first few characters only
                if len(matched_text) > 8:
                    return f"{matched_text[:4]}...{matched_text[-4:]}"
                return f"[{pii_type}]"

            elif pii_type == "PASSPORT":
                # Show last 3 characters
                if len(matched_text) >= 3:
                    return f"***{matched_text[-3:]}"
                return f"[{pii_type}]"

            elif pii_type == "DRIVERS_LICENSE":
                # Show last 4 characters
                if len(matched_text) >= 4:
                    return f"***{matched_text[-4:]}"
                return f"[{pii_type}]"

            elif pii_type == "TAX_ID":
                # Show last 4 digits
                digits = re.sub(r'[^0-9]', '', matched_text)
                if len(digits) >= 4:
                    return f"***-**-{digits[-4:]}"
                return f"[{pii_type}]"

            elif pii_type == "BANK_ACCOUNT":
                # Show last 4 digits
                if len(matched_text) >= 4:
                    return f"****{matched_text[-4:]}"
                return f"[{pii_type}]"

            elif pii_type == "IBAN":
                # Show country code and last 4 alnum chars. Formatted IBANs
                # can contain spaces/dashes, so strip those before slicing to
                # avoid picking up whitespace as the "last 4".
                clean = re.sub(r'[\s-]', '', matched_text)
                if len(clean) >= 6:
                    return f"{clean[:2]}****{clean[-4:]}"
                return f"[{pii_type}]"

            elif pii_type == "ADDRESS":
                # Just mask the whole address
                return "[ADDRESS REDACTED]"

            elif pii_type == "DOB":
                # Show just the year
                if len(matched_text) >= 4:
                    # Try to extract year
                    year_match = re.search(r'(19|20)\d{2}', matched_text)
                    if year_match:
                        return f"**/**/****"
                return f"[{pii_type}]"

            elif pii_type == "MAC_ADDRESS":
                # Show last segment only
                parts = re.split(r'[:-]', matched_text)
                if len(parts) >= 2:
                    return f"**:**:**:**:**:{parts[-1]}"
                return f"[{pii_type}]"

            elif pii_type == "VIN":
                # Show last 4 characters (serial number)
                if len(matched_text) >= 4:
                    return f"*************{matched_text[-4:]}"
                return f"[{pii_type}]"

            elif pii_type == "LICENSE_PLATE":
                # Show last 2 characters
                cleaned = matched_text.replace(' ', '').replace('-', '')
                if len(cleaned) >= 2:
                    return f"***{cleaned[-2:]}"
                return f"[{pii_type}]"

        # Default: full redaction
        return f"[{pii_type}]"

    def _redact_pii(self, text: str, spans: List[Tuple[int, int, str, str, str]]) -> str:
        """
        Redact PII from text.

        Args:
            text: Original text
            spans: List of (start, end, type, matched_text, config_type) tuples

        Returns:
            Text with PII redacted
        """
        if not spans:
            return text

        # Sort spans by start position in reverse to replace from end to start
        sorted_spans = sorted(spans, key=lambda x: x[0], reverse=True)

        result = text
        for start, end, pii_type, matched_text, config_type in sorted_spans:
            # Only redact if the type action is REDACT (not BLOCK)
            if self._get_type_action(config_type) == 'REDACT':
                replacement = self._get_redaction_replacement(pii_type, matched_text)
                result = result[:start] + replacement + result[end:]

        return result
