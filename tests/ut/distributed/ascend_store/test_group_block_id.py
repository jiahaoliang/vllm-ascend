# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.group_block_id import (
    _decode_group_block_id,
    _encode_group_block_id,
)


@pytest.mark.parametrize(
    ("group_id", "block_id", "encoded"),
    [
        (0, 0, -1),
        (0, 4_294_967_295, -4_294_967_296),
        (1, 0, -4_294_967_297),
        (4_294_967_295, 4_294_967_295, -18_446_744_073_709_551_616),
    ],
)
def test_group_block_id_encoding_has_stable_uint32_layout(
    group_id: int,
    block_id: int,
    encoded: int,
):
    assert _encode_group_block_id(group_id, block_id) == encoded
    assert _decode_group_block_id(encoded) == (group_id, block_id)


@pytest.mark.parametrize(
    ("group_id", "block_id"),
    [
        (-1, 0),
        (4_294_967_296, 0),
        (0, -1),
        (0, 4_294_967_296),
    ],
)
def test_group_block_id_encoding_rejects_values_outside_uint32(
    group_id: int,
    block_id: int,
):
    with pytest.raises(ValueError, match="uint32"):
        _encode_group_block_id(group_id, block_id)


@pytest.mark.parametrize("encoded", [0, 1, -18_446_744_073_709_551_617])
def test_group_block_id_decoding_rejects_values_outside_encoded_domain(
    encoded: int,
):
    with pytest.raises(ValueError, match="encoded group/block ID"):
        _decode_group_block_id(encoded)
