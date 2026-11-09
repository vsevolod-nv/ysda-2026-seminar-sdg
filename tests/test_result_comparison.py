import pytest

from seminar.metrics import matched_iou
from seminar.results import time_outliers, quality_report


def test_matched_iou_penalizes_missing_and_duplicate_boxes():
    box = [0, 0, 1, 1]
    assert matched_iou([box], [box]) == 1
    assert matched_iou([box, box], [box]) == .5
    assert matched_iou([box], [box, box]) == .5
    assert matched_iou([], [box]) == 0
    assert matched_iou([], []) == 1
    assert matched_iou([[0, 0, .5, 1]], [box]) == pytest.approx(.5)


def test_matching_finds_optimum_instead_of_greedy_pair():
    predicted = [[0, 0, .7, 1], [0, 0, .3, 1]]
    reference = [[0, 0, .5, 1], [.4, 0, 1, 1]]
    assert matched_iou(predicted, reference) == pytest.approx((.3 + .6) / 2)


def test_time_filter_uses_separate_method_distributions():
    rows = [dict(method=method, seconds=t * scale)
            for method, scale in [('manual', 10), ('assisted', 1)]
            for t in [8, 9, 10, 11, 12] * 10 + [.001, 100000]]
    frame = time_outliers(rows)
    assert frame.groupby('method').time_outlier.sum().to_dict() == {'assisted': 2, 'manual': 2}


def test_quality_without_controls_keeps_failed_honeypot_votes():
    candidate = dict(candidate_id='one', method='manual', image_id='image',
                     worker_id='author', status='ACCEPTED', reward=1, seconds=10)
    votes = []
    for i in range(3):
        for key in ['one', 'control_test']:
            votes.append(dict(assignment_id=str(i), worker_id=str(i), status='ACCEPTED', reward=.3,
                              input_values={'candidate_id': key},
                              output_values=dict(complete='yes', no_extra='yes', accurate='yes')))
    detail, summary = quality_report([candidate], votes)
    assert detail.iloc[0].reviewers == 3
    assert summary.iloc[0].pass_rate == 1
