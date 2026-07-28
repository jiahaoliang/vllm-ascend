# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

_UINT32_BITS = 32
_UINT32_LIMIT = 1 << _UINT32_BITS
_ENCODED_PAYLOAD_LIMIT = 1 << (2 * _UINT32_BITS)


def _encode_group_block_id(group_id: int, block_id: int) -> int:
    """Encode a pair of uint32 IDs into the private negative-ID domain."""
    if not 0 <= group_id < _UINT32_LIMIT or not 0 <= block_id < _UINT32_LIMIT:
        raise ValueError("group_id and block_id must be uint32 values")
    payload = (group_id << _UINT32_BITS) | block_id
    return -(payload + 1)


def _decode_group_block_id(encoded_id: int) -> tuple[int, int]:
    """Decode a private negative group/block ID."""
    payload = -encoded_id - 1
    if encoded_id >= 0 or payload >= _ENCODED_PAYLOAD_LIMIT:
        raise ValueError("invalid encoded group/block ID")
    return payload >> _UINT32_BITS, payload & (_UINT32_LIMIT - 1)
