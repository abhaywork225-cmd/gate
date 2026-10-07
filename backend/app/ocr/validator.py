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
        # Accept only characters actually read. Formatting delimiters and a
        # printed check-digit box may be removed; missing digits, owner letters
        # and brand-derived identities must never be invented.
        pattern = re.compile(
            r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ])'
            r'((?:[\s-]*[0-9]){6})[\s-]*[\[(|]?\s*([0-9])[\])|]?'
            r'(?![A-Z0-9])'
        )
        candidates = {normalize(''.join(m.groups())) for m in pattern.finditer(raw.upper())}
        if len(candidates) > 1:
            return Validation(raw, '', read.confidence, False, False, 'AMBIGUOUS')
        if candidates:
            normalized = next(iter(candidates))
        else:
            size_match = SIZE_CODE_PATTERN.search(raw)
            if size_match and sum(c.isalpha() for c in normalize(raw)) <= 2:
                iso_code = ''.join(size_match.groups()).upper()
                return Validation(raw, iso_code, read.confidence, True, False,
                                  'VALID_SIZE_CODE', size_type=iso_code)
            normalized = normalize(raw)[:256]
        valid_format = bool(STRUCTURE.fullmatch(normalized))
        valid_digit = valid_format and check_digit(normalized[:10]) == int(normalized[10])

        # Reject known hallucinations from old prompt examples
        if normalized in ('TIIU5077669', 'MSCU6546021'):
            return Validation(raw, '', 0.0, False, False, 'PROMPT_LEAK_DISCARDED')

        status = 'VALID' if valid_digit else 'INVALID_CHECK_DIGIT' if valid_format else 'INVALID_FORMAT'
        iso_code, human_size = extract_iso_size_code(raw)
        return Validation(raw, normalized, read.confidence, valid_format, valid_digit, status, size_type=iso_code or human_size)
