import numpy as np
import pytest

from laia.data.sampler import BalancedPaddingSampler, SortedSampler


@pytest.mark.parametrize("N", [1, 11, 100])
@pytest.mark.parametrize("batch_size", [1, 5, 8])
@pytest.mark.parametrize("C", [1, 3])
@pytest.mark.parametrize("H", [1, 101, 333])
@pytest.mark.parametrize("W", [1, 101, 333])
def test_sorted_sampler(N, batch_size, C, H, W):
    value_fn = lambda x: x.size
    data = [np.random.randint(1, 100, size=(C, H, W)) for _ in range(N)]
    bs = SortedSampler(data, value_fn, reverse=True, batch_size=batch_size)

    # check when not enough samples to bucket
    if N < batch_size:
        assert len(bs.batches) == 1
        assert len(bs.batches[0]) == N

    # check no values missing
    expected = sorted(range(len(data)), key=lambda i: value_fn(data[i]))
    assert expected == [x for b in bs.batches for x in b]

    # check batch_size correct
    for i, b in enumerate(bs.batches):
        if i == len(bs.batches) - 1 and len(b) != batch_size:
            # leftover samples
            assert len(b) == N % batch_size
        else:
            # base case
            assert len(b) == batch_size

    # check len
    num_batches, leftover = divmod(len(data), batch_size)
    assert len(bs) == num_batches + bool(leftover)


@pytest.mark.parametrize("N", [1, 11, 100])
@pytest.mark.parametrize("batch_size", [1, 5, 8])
@pytest.mark.parametrize("C", [1])
@pytest.mark.parametrize("H", [1, 101])
@pytest.mark.parametrize("W", [1, 101])
def test_sorted_sampler_drop_last(N, batch_size, C, H, W):
    value_fn = lambda x: -x.size
    data = [np.random.randint(1, 100, size=(C, H, W)) for _ in range(N)]
    bs = SortedSampler(data, value_fn, batch_size=batch_size, drop_last=True)

    expected = sorted(range(len(data)), key=lambda i: value_fn(data[i]))
    if N % batch_size:
        # delete leftover
        del expected[-(N % batch_size) :]

    if N < batch_size:
        # not enough samples to bucket and got deleted.
        assert len(bs.batches) == 0
        max_size = 0
    else:
        max_size = value_fn(data[expected[0]]) * batch_size

    # check no values missing
    assert expected == [x for b in bs.batches for x in b]

    # check batch_size correct
    assert all(len(b) == batch_size for b in bs.batches)

    # check len
    assert len(bs) == len(data) // batch_size


@pytest.mark.parametrize("N", [1, 11, 100])
@pytest.mark.parametrize("batch_size", [1, 5, 8])
@pytest.mark.parametrize("C", [1, 3])
@pytest.mark.parametrize("H", [1, 101, 333])
@pytest.mark.parametrize("W", [1, 101, 333])
def test_balanced_sampler(N, batch_size, C, H, W):
    value_fn = lambda x: x.size
    data = [np.random.randint(1, 100, size=(C, H, W)) for _ in range(N)]
    bs = BalancedPaddingSampler(data, value_fn, reverse=True, batch_size=batch_size)

    # check when not enough samples to bucket
    if N < batch_size:
        assert len(bs.batches) == 1
        assert len(bs.batches[0]) == N

    # check no values missing
    expected = sorted(range(len(data)), key=lambda i: value_fn(data[i]))
    assert expected == [x for b in bs.batches for x in b]

    # check all batches <= max_size
    max_size = value_fn(data[expected[0]]) * batch_size
    assert all(value_fn(data[b[0]]) * len(b) <= max_size for b in bs.batches)


def test_balanced_sampler_specific_case():
    data = [2, 2, 5, 5, 3, 2, 4, 9, 5, 1, 6, 4, 9, 4]
    batches = BalancedPaddingSampler.prepare_batches(
        data, lambda x: x, reverse=True, batch_size=3
    )
    expected = [[9, 9, 6], [5, 5, 5, 4, 4], [4, 3, 2, 2, 2, 1]]
    assert [[data[i] for i in b] for b in batches] == expected

    data = [20, 10, 10, 5, 5, 5, 5]
    batches = BalancedPaddingSampler.prepare_batches(
        data, lambda x: x, reverse=True, batch_size=1
    )
    expected = [[20], [10, 10], [5, 5, 5, 5]]
    assert [[data[i] for i in b] for b in batches] == expected
