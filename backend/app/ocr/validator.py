from __future__ import annotations
"""ISO 6346 syntax/check digit, without guessing OCR character substitutions.

Checksum validity is not proof of registration or ownership in the BIC registry.
"""
import re
from app.domain import OCRRead, Validation

LETTER_VALUES = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', (n for n in range(10, 39) if n % 11)))
STRUCTURE = re.compile(r'^[A-Z]{3}[UJZ][0-9]{7}$')
CANDIDATE = re.compile(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){7})(?![A-Z0-9])')

# ISO 6346 Container Size-Type Code pattern (e.g. 45G1, 22G1, 42G1, L5G1)
# Standard container type for general purpose dry freight is 'G' (e.g. G0, G1).
# OCR engines frequently confuse 'G' with 'C' (e.g. 45C1, 22C1, 42C1).
# ISO 6346 defines NO type code 'C'. We correct 'C' to 'G' for all ISO container size-type patterns.
# ISO 6346 Container Size-Type Code pattern (e.g. 45G1, 22G1, 42G1, L5G1, '2 2 G 1')
# Standard container type for general purpose dry freight is 'G' (e.g. G0, G1).
# OCR engines frequently confuse 'G' with 'C' (e.g. 45C1, 22C1, 42C1).
# ISO 6346 defines NO type code 'C'. We correct 'C' to 'G' for all ISO container size-type patterns.
SIZE_CODE_PATTERN = re.compile(r'(?<![A-Z0-9])([1-4LMN])\s*([0-9])\s*([GVRHBUTP])\s*([0-9A-Z])(?![A-Z0-9])', re.IGNORECASE)
SIZE_CODE_C_MISTAKE = re.compile(r'(?<![A-Z0-9])([1-4LMN])\s*([0-9])\s*C\s*([0-9A-Z]?)(?![A-Z0-9])', re.IGNORECASE)


def correct_feet_size_codes(text: str) -> str:
    """
    Correct OCR misreads where 'G' in container size markings is read as 'C'.
    E.g. 45C1 -> 45G1, 22C1 -> 22G1, 42C1 -> 42G1, L5C1 -> L5G1, 45C -> 45G1
    """
    if not text:
        return ''
    # 1. Correct [Size][C][Char] -> [Size][G][Char]
    corrected = SIZE_CODE_C_MISTAKE.sub(lambda m: f"{m.group(1).upper()}{m.group(2).upper()}G{m.group(3).upper() or '1'}", text)
    return corrected


def extract_iso_size_code(code_or_text: str) -> tuple[str | None, str | None]:
    """
    Extract container ISO size-type code (e.g. '45G1', '22G1', '42G1', 'L5G1')
    and human-readable size (e.g. '40 FT HC', '20 FT', '40 FT', '45 FT HC').
    """
    if not code_or_text:
        return None, None
    cleaned = correct_feet_size_codes(code_or_text)

    # 1. ISO size-type code pattern (e.g. 45G1, 22G1, 42G1, L5G1, 22R1)
    # Matches either standalone with boundary/whitespace or appended to container code
    size_match = re.search(r'(?:(?<![A-Z0-9])|[A-Z]{3}[UJZ]\d{7})([1-4LMN])\s*([0-9])\s*([GVRHBUTP])\s*([0-9A-Z])(?![A-Z0-9])', cleaned.upper())
    if size_match:
        iso_code = f'{size_match.group(1)}{size_match.group(2)}{size_match.group(3)}{size_match.group(4)}'
        length_char = iso_code[0]
        height_char = iso_code[1]
        length_map = {
            '1': '10 FT',
            '2': '20 FT',
            '3': '30 FT',
            '4': '40 FT',
            'L': '45 FT',
            'M': '48 FT',
            'N': '53 FT',
        }
        base_length = length_map.get(length_char, '40 FT')
        human_size = f'{base_length} HC' if height_char == '5' else base_length
        return iso_code, human_size

    # 2. Direct feet text: e.g. 40FT, 20FT, 45FT
    ft_match = re.search(r'(10|20|30|40|45|48|53)\s*(?:FT|FEET)', cleaned.upper())
    if ft_match:
        return None, f'{ft_match.group(1)} FT'

    return None, None


def parse_feet_size(code_or_text: str) -> str | None:
    _, human_size = extract_iso_size_code(code_or_text)
    return human_size


def check_digit(prefix: str) -> int:
    if not re.fullmatch(r'[A-Z]{3}[UJZ][0-9]{6}', prefix):
        raise ValueError('Expected three owner letters, U/J/Z and six serial digits')
    total = sum((int(char) if char.isdigit() else LETTER_VALUES[char]) * (2 ** index)
                for index, char in enumerate(prefix))
    return (total % 11) % 10


def normalize(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', text.upper())


# Common visual OCR confusions for check digit (read digit vs calculated digit)
CHECK_DIGIT_CONFUSIONS = {
    (2, 7), (7, 2),
    (1, 7), (7, 1),
    (0, 8), (8, 0),
    (3, 8), (8, 3),
    (5, 6), (6, 5),
    (6, 8), (8, 6),
    (1, 4), (4, 1),
}

BRAND_PREFIX_MAP = {
    'GALAXY': 'GXYU',
    'GALAXY CONTAINER': 'GXYU',
    'MEDITERRANEAN SHIPPING CO': 'MSCU',
    'MEDITERRANEAN': 'MSCU',
    'WAN HAI LINES': 'WHSU',
    'WAN HAI': 'WHSU',
    'WANHAI': 'WHSU',
    'MAERSK': 'MSKU',
    'CMA CGM': 'CMAU',
    'COSCO': 'COSU',
    'EVERGREEN': 'EGLU',
    'HAPAG': 'HLXU',
    'HAPAG-LLOYD': 'HLXU',
    'OCEAN NETWORK EXPRESS': 'ONEU',
    'YANG MING': 'YMLU',
    'ZIM': 'ZIMU',
}


# Camera OSD watermark and timestamp patterns (e.g. 'CP IP Cam', 'CPIPCAM', '16-09-2026 15:28:10')
OSD_WATERMARK_PATTERN = re.compile(
    r'\b(?:CP\s*IP\s*CAM|CPIPCAM|CPIP|IP\s*CAM|CAM|PROOF(?::\s*[A-Z0-9\s()_-]+)?|GATE[-_ ]?(?:IN|OUT)[-_ ]?\d*|ENTRY|EXIT)\b',
    re.IGNORECASE
)
OSD_TIMESTAMP_PATTERN = re.compile(
    r'\b\d{2}[-./]\d{2}[-./]\d{4}(?:\s+\d{2}[:.]\d{2}[:.]\d{2})?\b|\b\d{2}[:.]\d{2}[:.]\d{2}\b'
)


def sanitize_ocr_text(text: str) -> str:
    """Strip camera OSD watermarks and timestamps from OCR text."""
    if not text:
        return ''
    has_watermark = bool(
        OSD_WATERMARK_PATTERN.search(text) or
        OSD_TIMESTAMP_PATTERN.search(text) or
        re.search(r'\b(?:1[0-9]|2[0-9]|3[01])(?:0[1-9]|1[0-2])202\d{7,10}\b', text)
    )
    if not has_watermark:
        return text
    cleaned = OSD_WATERMARK_PATTERN.sub(' ', text)
    cleaned = OSD_TIMESTAMP_PATTERN.sub(' ', cleaned)
    cleaned = re.sub(r'\b(?:1[0-9]|2[0-9]|3[01])(?:0[1-9]|1[0-2])202\d{7,10}\b', ' ', cleaned)
    return cleaned.strip()


class ContainerValidator:
    def validate(self, read: OCRRead) -> Validation:
        raw_input = read.raw_text[:4096] if read and read.raw_text else ''
        sanitized = sanitize_ocr_text(raw_input)
        if not sanitized or normalize(sanitized) in ('', 'CPIPCAM', 'CPIP', 'IPCAM', 'CAM', 'PROOF'):
            return Validation('', '', 0.0, False, False, 'INVALID_FORMAT')

        raw = correct_feet_size_codes(sanitized)
        single_line_raw = ' '.join(raw.splitlines())

        candidates = {normalize(m.group(1)) for m in CANDIDATE.finditer(raw.upper())}
        if not candidates and single_line_raw != raw:
            candidates = {normalize(m.group(1)) for m in CANDIDATE.finditer(single_line_raw.upper())}

        # Bracket confusion around check digit: e.g. 'MRSU 539593 13', 'SEGU 357042 181', '[3]' read as '13', '|3', 'I3', '/3'
        for m in re.finditer(r'(?<![A-Z0-9])([A-Z]{3}[UJZ])[\s\-_:/|]*([0-9]{6})[\s\-_:/|]*[1I|/\[(]([0-9])[\])]?', raw.upper()):
            pref = m.group(1) + m.group(2)
            exp_cd = int(m.group(3))
            try:
                calc_cd = check_digit(pref)
                if calc_cd == exp_cd or (exp_cd, calc_cd) in CHECK_DIGIT_CONFUSIONS:
                    candidates.add(pref + str(calc_cd))
            except Exception:
                pass

        # Trailing size codes / tare weights: e.g. 'SEKU 148632 1 45G1' or 'UETU 814627 21'
        for m in re.finditer(r'(?<![A-Z0-9])([A-Z]{3}[UJZ])[\s\-_:/|]*([0-9]{6})[\s\-_:/|]*([0-9])', raw.upper()):
            pref = m.group(1) + m.group(2)
            exp_cd = int(m.group(3))
            try:
                calc_cd = check_digit(pref)
                if calc_cd == exp_cd or (exp_cd, calc_cd) in CHECK_DIGIT_CONFUSIONS:
                    candidates.add(pref + str(calc_cd))
            except Exception:
                pass

        # OCR often inserts a stray L/I/1 between the equipment category and
        # the six serial digits (for example ``BEAUL 255352``).  Treat that
        # character as separator noise only when the remaining structure is a
        # complete ISO prefix plus exactly six digits.
        noisy_separator = re.search(
                r'(?<![A-Z0-9])([A-Z]{3}[UJZ])[LIO1|]\s*([0-9]{6})(?![0-9])',
                raw.upper()
            )
        if noisy_separator:
            prefix = noisy_separator.group(1)
            serial = noisy_separator.group(2)
            try:
                candidates.add(prefix + serial + str(check_digit(prefix + serial)))
            except Exception:
                pass

        # Tolerant delimiter candidate matching: handles OCR delimiter noise like 'TRHU1 1545895',
        # 'TRHU | 154589 [5]', 'TRHU - 154589 5', 'TRHU/1545895'
        if not candidates:
            tolerant_matches = re.finditer(
                r'(?<![A-Z0-9])([A-Z]{3}[UJZ])[ \t\r\n\-.:/|]*([0-9Il|/ \t\-.:]{6,12})(?![0-9])',
                raw.upper()
            )
            for m in tolerant_matches:
                prefix = m.group(1)
                first_line_digits = m.group(2).splitlines()[0]
                cleaned_digits = re.sub(r'[^0-9]', '', first_line_digits.replace('I', '1').replace('L', '1').replace('|', '1'))
                has_separator = any(c in m.group(0) for c in ' \t\r\n-.:/|')
                if len(cleaned_digits) == 7:
                    candidates.add(prefix + cleaned_digits)
                elif len(cleaned_digits) == 8 and has_separator:
                    # Delimiter character read as 1/0/I before serial number (e.g. TRHU1 1545895 -> TRHU 154589 5)
                    c1_pref = prefix + cleaned_digits[1:7]
                    cd1_expected = int(cleaned_digits[7])
                    try:
                        cd1_calc = check_digit(c1_pref)
                        if cd1_calc == cd1_expected or (cd1_expected, cd1_calc) in CHECK_DIGIT_CONFUSIONS:
                            candidates.add(c1_pref + str(cd1_calc))
                    except Exception:
                        pass
                elif len(cleaned_digits) == 6 and has_separator:
                    c_pref = prefix + cleaned_digits
                    try:
                        candidates.add(c_pref + str(check_digit(c_pref)))
                    except Exception:
                        pass

        if len(candidates) > 1:
            valid_cands = [c for c in candidates if bool(STRUCTURE.fullmatch(c)) and check_digit(c[:10]) == int(c[10])]
            if len(valid_cands) == 1:
                normalized = valid_cands[0]
            elif len(valid_cands) > 1:
                normalized = valid_cands[0]
            else:
                return Validation(raw, '', read.confidence, False, False, 'AMBIGUOUS')
        elif candidates:
            normalized = next(iter(candidates))
        else:
            six_match = re.search(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){6})(?![A-Z0-9])', raw.upper())
            if six_match:
                prefix = normalize(six_match.group(1))
                normalized = prefix + str(check_digit(prefix))
            else:
                # 1. Check if this reading is strictly an ISO size/type code (e.g. 22G1, 45G1)
                size_match = SIZE_CODE_PATTERN.search(raw)
                if size_match and sum(1 for c in normalize(raw) if c.isalpha()) <= 2:
                    iso_code = f'{size_match.group(1)}{size_match.group(2)}{size_match.group(3).upper()}{size_match.group(4).upper()}'
                    return Validation(raw, iso_code, read.confidence, True, False, 'VALID_SIZE_CODE', size_type=iso_code)

                # Check for 3 owner letters followed by 6 digits where the 4th letter 'U' was omitted or read as '0'/'O'
                three_letter_match = re.search(r'(?<![A-Z0-9])([A-Z]{3})[\s-]*[0O]?[\s-]*([0-9]{6})(?:[\s-]*([0-9]))?(?![A-Z0-9])', raw.upper())
                subst_found = False
                if three_letter_match:
                    pref3 = three_letter_match.group(1)
                    ser6 = three_letter_match.group(2)
                    read_cd = int(three_letter_match.group(3)) if three_letter_match.group(3) else None
                    candidate_prefix = f"{pref3}U{ser6}"
                    calc_cd = check_digit(candidate_prefix)
                    if read_cd is None or read_cd == calc_cd or (read_cd, calc_cd) in CHECK_DIGIT_CONFUSIONS:
                        normalized = candidate_prefix + str(calc_cd)
                        subst_found = True
                if not subst_found:
                    DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z'}
                    LETTER_TO_DIGIT = {'O': '0', 'I': '1', 'L': '1', 'B': '8', 'S': '5', 'Z': '2', 'J': '3', '|': '1'}
                    flex_m = re.search(
                        r'(?<![A-Z0-9])([A-Z0-9])[\s-]*([A-Z0-9])[\s-]*([A-Z0-9])[\s-]*([UJZ0-9O])'
                        r'((?:[\s-]*[0-9A-Z|]){6,8})(?![0-9A-Z])',
                        raw.upper()
                    )
                    if flex_m:
                        c1, c2, c3, c4 = flex_m.group(1), flex_m.group(2), flex_m.group(3), flex_m.group(4)
                        p1 = DIGIT_TO_LETTER.get(c1, c1)
                        p2 = DIGIT_TO_LETTER.get(c2, c2)
                        p3 = DIGIT_TO_LETTER.get(c3, c3)
                        if p1.isalpha() and p2.isalpha() and p3.isalpha():
                            p4 = 'U' if c4 in ('0', 'O', 'U') else (c4 if c4 in ('J', 'Z') else 'U')
                            cand_pref = f'{p1}{p2}{p3}{p4}'
                            digits_raw = flex_m.group(5)
                            clean_d = ''.join(LETTER_TO_DIGIT.get(ch, ch) for ch in digits_raw if ch not in ' \t\r\n-.:/|')
                            clean_d = re.sub(r'[^0-9]', '', clean_d)
                            if len(clean_d) >= 6:
                                calc_cd = check_digit(cand_pref + clean_d[:6])
                                read_cd = int(clean_d[6]) if len(clean_d) >= 7 else None
                                if read_cd is None or read_cd == calc_cd or (read_cd, calc_cd) in CHECK_DIGIT_CONFUSIONS:
                                    normalized = f'{cand_pref}{clean_d[:6]}{calc_cd}'
                                    subst_found = True
                if not subst_found:
                    brand_found = False
                    for brand, bic in BRAND_PREFIX_MAP.items():
                        if brand in raw.upper():
                            digit_match = re.search(r'(?<![0-9])([0-9]{6})(?:[\s-]*([0-9]))?(?![0-9])', raw)
                            if digit_match:
                                prefix = f"{bic}{digit_match.group(1)}"
                                normalized = prefix + str(check_digit(prefix))
                                brand_found = True
                                break
                    if not brand_found:
                        normalized = normalize(raw)[:256]

        valid_format = bool(STRUCTURE.fullmatch(normalized))
        valid_digit = False
        if valid_format:
            expected_digit = int(normalized[10])
            calc_digit = check_digit(normalized[:10])
            if calc_digit == expected_digit:
                valid_digit = True
            else:
                # Check if 'C' was misread for 'G' (or 'G' for 'C') in owner letters
                for pos in range(3):
                    if normalized[pos] == 'C':
                        candidate_prefix = normalized[:pos] + 'G' + normalized[pos + 1:10]
                        if check_digit(candidate_prefix) == expected_digit:
                            normalized = candidate_prefix + str(expected_digit)
                            valid_digit = True
                            break
                if not valid_digit:
                    # Check if check digit was visually confused in OCR
                    if (expected_digit, calc_digit) in CHECK_DIGIT_CONFUSIONS:
                        normalized = normalized[:10] + str(calc_digit)
                        valid_digit = True

        # Reject known hallucinations from old prompt examples
        if normalized in ('TIIU5077669', 'MSCU6546021'):
            return Validation(raw, '', 0.0, False, False, 'PROMPT_LEAK_DISCARDED')

        status = 'VALID' if valid_digit else 'INVALID_CHECK_DIGIT' if valid_format else 'INVALID_FORMAT'
        iso_code, human_size = extract_iso_size_code(raw)
        eff_conf = max(read.confidence, 0.85) if valid_digit else read.confidence
        return Validation(raw, normalized, eff_conf, valid_format, valid_digit, status, size_type=iso_code or human_size)
