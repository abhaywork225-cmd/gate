import pytest
from app.domain import OCRRead
from app.ocr.validator import ContainerValidator, check_digit, normalize


@pytest.mark.parametrize('text', ['CSQU3054383', 'csqu 305438 3', 'CSQU-305438-3', 'ID: CSQU\n305438\n3'])
def test_valid_container(text):
    value = ContainerValidator().validate(OCRRead(text, .94))
    assert value.raw_text == text
    assert value.normalized_text == 'CSQU3054383'
    assert value.valid_format and value.valid_check_digit


def test_missing_check_digit_is_not_invented():
    val = ContainerValidator().validate(OCRRead('GXYU 509070', .95))
    assert val.normalized_text == 'GXYU509070'
    assert not val.valid_format and not val.valid_check_digit


@pytest.mark.parametrize('text,status', [('CSQU3054384', 'INVALID_CHECK_DIGIT'), ('CSQA3054383', 'INVALID_FORMAT'),
                                       ('CSQU3O54383', 'INVALID_FORMAT'), ('CSQU30543833', 'INVALID_FORMAT'),
                                       ('No number visible', 'INVALID_FORMAT'), ('', 'INVALID_FORMAT')])
def test_invalid_ocr_is_not_trusted(text, status):
    result = ContainerValidator().validate(OCRRead(text, .99))
    assert result.validation_status == status
    assert not result.valid_check_digit


def test_ambiguous_ids_and_checksum_algorithm():
    assert check_digit('CSQU305438') == 3
    assert check_digit('MSCU663987') == 0
    result = ContainerValidator().validate(OCRRead('CSQU3054383 MSCU6639870', .9))
    assert result.validation_status == 'AMBIGUOUS'
    assert normalize(' abcu-123456-7 ') == 'ABCU1234567'
    with pytest.raises(ValueError):
        check_digit('INVALID')


def test_feet_size_code_c_to_g_correction():
    from app.ocr.validator import correct_feet_size_codes, parse_feet_size
    assert correct_feet_size_codes('45C1') == '45G1'
    assert correct_feet_size_codes('22C1') == '22G1'
    assert correct_feet_size_codes('42C1') == '42G1'
    assert correct_feet_size_codes('L5C1') == 'L5G1'
    assert correct_feet_size_codes('45C') == '45G1'
    assert correct_feet_size_codes('42 C 1') == '42G1'

    assert parse_feet_size('45C1') == '40 FT HC'
    assert parse_feet_size('45G1') == '40 FT HC'
    assert parse_feet_size('42G1') == '40 FT'
    assert parse_feet_size('22G1') == '20 FT'
    assert parse_feet_size('L5G1') == '45 FT HC'
    assert parse_feet_size('40 FT') == '40 FT'
    assert parse_feet_size('20FEET') == '20 FT'

    # Validator normalizes 45C1 directly to 45G1
    val = ContainerValidator().validate(OCRRead('45C1', .95))
    assert val.normalized_text == '45G1'
    assert val.validation_status == 'VALID_SIZE_CODE'


def test_inserted_digit_is_not_silently_removed():
    val = ContainerValidator().validate(OCRRead('TRHU1 1545895', .95))
    assert not val.valid_check_digit


@pytest.mark.parametrize('raw', ['CSQU 305438 [3]', 'CSQU 305438 (3)', 'CSQU 305438 |3|'])
def test_printed_check_digit_delimiters(raw):
    val = ContainerValidator().validate(OCRRead(raw, .91))
    assert val.normalized_text == 'CSQU3054383'
    assert val.valid_check_digit
    assert val.confidence == .91


@pytest.mark.parametrize('raw', ['BEAUL 255352', 'MAERSK 305438', 'CSQ 305438 3', 'CSQU 305438 8'])
def test_no_identity_guessing(raw):
    assert not ContainerValidator().validate(OCRRead(raw, .99)).valid_check_digit
