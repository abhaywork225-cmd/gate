"""Regression cases for field OCR failures, without fabricated confidence."""
from pathlib import Path
import os
import cv2
import numpy as np
import pytest
from app.ocr.container_ocr import rectify_printed_digit_box, vertical_glyph_column


def test_rectify_slanted_printed_box_preserves_glyph_and_input():
    image = np.full((120,80,3), 60, np.uint8)
    corners = np.array([[23,8],[66,15],[49,110],[6,103]])
    cv2.polylines(image, [corners], True, (240,240,240), 3)
    cv2.line(image, (42,30), (29,89), (240,240,240), 5)
    before = image.copy()
    rectified = rectify_printed_digit_box(image)
    assert rectified is not None
    assert rectified.shape[0] > rectified.shape[1]
    assert (rectified.max(axis=2)>150).sum() > 50
    assert np.array_equal(image,before)


@pytest.mark.parametrize('digit', ['1','7','0'])
def test_unboxed_digit_is_not_treated_as_printed_rectangle(digit):
    image = np.full((100,70,3), 60, np.uint8)
    cv2.putText(image,digit,(10,80),cv2.FONT_HERSHEY_SIMPLEX,2.5,(240,240,240),4)
    assert rectify_printed_digit_box(image) is None


def test_vertical_column_excludes_adjacent_iso_markings():
    number = [(200-15*i,10+80*i,25,65) for i in range(11)]
    iso = [(300,200+80*i,30,65) for i in range(4)]
    assert vertical_glyph_column(number+iso,900) == number
    # Missing glyphs cannot be supplied by another column or by the checksum.
    assert vertical_glyph_column(number[:5]+number[6:]+iso,900) is None


def test_two_complete_columns_are_ambiguous():
    first = [(40,10+80*i,25,65) for i in range(11)]
    second = [(180,10+80*i,25,65) for i in range(11)]
    assert vertical_glyph_column(first+second,900) is None


@pytest.mark.skipif(os.environ.get('GATE_GPU_REPLAY') != '1', reason='Explicit local GPU/model replay')
@pytest.mark.parametrize('filename,number,minimum', [('hlhu_boxed_digit.png','HLHU8516751',.9),
                                                     ('oneu_vertical.png','ONEU7355689',.5),
                                                     ('hlhu_door_bar.png','HLHU8516751',.9)])
def test_real_number_crops_read_complete_number_on_gpu(filename,number,minimum):
    from app.config.settings import Settings
    from app.ocr.ocr_engine import EasyOCREngine
    from app.ocr.validator import ContainerValidator
    image = cv2.imread(str(Path(__file__).parent/'fixtures/ocr'/filename))
    engine = EasyOCREngine(Settings(ocr_gpu=True,ocr_download_enabled=False,prosper_enabled=False))
    result = ContainerValidator().validate(engine.read(image))
    assert result.normalized_text == number
    assert result.valid_check_digit and result.confidence >= minimum


def test_narrow_slanted_box_does_not_duplicate_corner_selection():
    image = np.full((100,45,3),60,np.uint8)
    corners = np.array([[20,5],[38,26],[23,90],[6,67]])
    cv2.polylines(image,[corners],True,(240,240,240),2)
    cv2.line(image,(24,30),(17,65),(240,240,240),2)
    assert rectify_printed_digit_box(image) is not None


def test_horizontal_layout_excludes_full_height_door_bar_but_not_glyphs():
    from app.ocr.container_ocr import horizontal_number_layout
    owner = [(10+10*i,15,8,20) for i in range(4)]
    serial = [(75+10*i,15,8,20) for i in range(6)]
    digit = [(145,13,12,24)]
    bar = [(57,0,6,48)]
    assert horizontal_number_layout(owner+bar+serial+digit,(48,168,3)) == (owner,serial,digit)
    assert horizontal_number_layout(owner[:3]+bar+serial+digit,(48,168,3)) is None
    lower_row = [(x,y+25,w,h) for x,y,w,h in serial]
    assert horizontal_number_layout(owner+lower_row+digit,(70,168,3)) is None



@pytest.mark.skipif(os.environ.get('GATE_GPU_REPLAY') != '1', reason='Explicit local GPU/model replay')
def test_blurry_crop_cannot_turn_into_different_checksum_valid_number():
    from app.config.settings import Settings
    from app.ocr.ocr_engine import EasyOCREngine
    from app.ocr.validator import ContainerValidator
    image = cv2.imread(str(Path(__file__).parent/'fixtures/ocr/hlhu_distant_negative.png'))
    engine = EasyOCREngine(Settings(ocr_gpu=True,ocr_download_enabled=False,prosper_enabled=False))
    result = ContainerValidator().validate(engine.read(image))
    assert not result.valid_check_digit or result.normalized_text == 'HLHU8516751'
