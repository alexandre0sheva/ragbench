from __future__ import annotations

import math

import pytest

from ragbench.evaluation.agreement import krippendorff_alpha

NAN = math.nan


def test_alpha_reproduces_krippendorffs_published_interval_example():
    # Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 coders x 12 units with missing values; interval alpha = 0.849.
    coders = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, NAN, NAN, NAN],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, NAN, 3],
        [NAN, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, NAN],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, NAN],
    ]
    units = [list(column) for column in zip(*coders, strict=True)]

    assert krippendorff_alpha(units) == pytest.approx(0.849, abs=5e-4)


def test_two_raters_disagreeing_on_one_unit_match_the_hand_computed_alpha():
    # values 1,2,3,4 | 1,2,3,5: D_o = 2/8, D_e = 222/56, alpha = 1 - 14/222
    assert krippendorff_alpha([[1, 1], [2, 2], [3, 3], [4, 5]]) == pytest.approx(1 - 14 / 222)


def test_perfect_agreement_is_one():
    assert krippendorff_alpha([[1, 1, 1], [3, 3, 3], [5, 5, NAN]]) == 1.0


def test_alpha_is_undefined_without_variation_or_without_pairable_units():
    assert krippendorff_alpha([[4, 4], [4, 4]]) is None  # every value the same: no expected disagreement to compare with
    assert krippendorff_alpha([[1, NAN], [NAN, 2]]) is None  # no unit has two values
    assert krippendorff_alpha([]) is None


def test_systematic_disagreement_goes_below_zero():
    assert krippendorff_alpha([[0, 5], [5, 0], [0, 5], [5, 0]]) < 0
