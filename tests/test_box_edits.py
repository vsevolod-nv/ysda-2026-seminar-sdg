from seminar.metrics import box_edit_counts


def test_box_edits_ignore_order_and_rounding():
    a = [[0, 0, .2, .2], [.5, .5, .8, .8]]
    counts = box_edit_counts(a, [[.50001, .5, .8, .8], a[0]])
    assert counts == dict(initial=2, unchanged=2, adjusted=0, removed=0, added=0)


def test_box_edits_account_for_edits_additions_deletions_and_duplicates():
    a = [[0, 0, .2, .2], [.5, .5, .8, .8]]
    b = [[.01, 0, .21, .2], [.8, 0, .9, .1]]
    assert box_edit_counts(a, b) == dict(initial=2, unchanged=0, adjusted=1, removed=1, added=1)
    assert box_edit_counts([a[0], a[0]], [a[0]])['removed'] == 1
    assert box_edit_counts([], b)['added'] == 2
    assert box_edit_counts(a, [])['removed'] == 2
    assert box_edit_counts([], []) == dict(initial=0, unchanged=0, adjusted=0, removed=0, added=0)
