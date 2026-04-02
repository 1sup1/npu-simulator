"""MAC 연산 엔진 및 패턴 판정 로직"""

from __future__ import annotations

import time
from typing import List, Tuple

# 표준 라벨
LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"

# 허용오차
EPSILON = 1e-9

# 성능 측정 반복 횟수
BENCHMARK_REPEATS = 10

Matrix = List[List[float]]


def mac(pattern: Matrix, filt: Matrix) -> float:
    """MAC(Multiply-Accumulate) 연산: 같은 위치의 값을 곱하고 모두 더한다.

    Args:
        pattern: n×n 입력 패턴
        filt: n×n 필터

    Returns:
        점수 (float)
    """
    total = 0.0
    for row_p, row_f in zip(pattern, filt):
        for val_p, val_f in zip(row_p, row_f):
            total += val_p * val_f
    return total


def mac_flat(pattern: Matrix, filt: Matrix) -> float:
    """1차원 변환 후 MAC 연산 (보너스: 메모리 접근 최적화).

    2차원 배열을 1차원으로 펼쳐 단일 루프로 처리한다.
    """
    flat_p = [v for row in pattern for v in row]
    flat_f = [v for row in filt for v in row]
    total = 0.0
    for vp, vf in zip(flat_p, flat_f):
        total += vp * vf
    return total


def normalize_label(raw: str) -> str:
    """입력 라벨을 표준 라벨로 정규화한다.

    '+', 'cross', 'Cross', 'CROSS' → 'Cross'
    'x', 'X'                       → 'X'
    """
    s = raw.strip().lower()
    if s in ("+", "cross"):
        return LABEL_CROSS
    if s == "x":
        return LABEL_X
    return raw.strip()


def judge(score_cross: float, score_x: float) -> str:
    """두 필터 점수를 비교하여 판정한다.

    Returns:
        'Cross', 'X', 또는 'UNDECIDED'
    """
    diff = abs(score_cross - score_x)
    if diff < EPSILON:
        return LABEL_UNDECIDED
    if score_cross > score_x:
        return LABEL_CROSS
    return LABEL_X


def benchmark_mac(pattern: Matrix, filt: Matrix,
                  repeats: int = BENCHMARK_REPEATS) -> float:
    """MAC 연산을 repeats 회 반복하여 평균 시간(ms)을 반환한다."""
    start = time.perf_counter()
    for _ in range(repeats):
        mac(pattern, filt)
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000  # ms


def validate_matrix(matrix: Matrix, expected_n: int) -> Tuple[bool, str]:
    """행렬 크기가 expected_n × expected_n 인지 검증한다."""
    if len(matrix) != expected_n:
        return False, f"행 수 불일치: {len(matrix)}행 (기대: {expected_n}행)"
    for i, row in enumerate(matrix):
        if len(row) != expected_n:
            return False, f"{i+1}행 열 수 불일치: {len(row)}열 (기대: {expected_n}열)"
    return True, ""


def generate_cross(n: int) -> Matrix:
    """n×n 십자가 패턴을 자동 생성한다 (보너스: 패턴 생성기)."""
    mid = n // 2
    grid = [[0.0] * n for _ in range(n)]
    for i in range(n):
        grid[mid][i] = 1.0
        grid[i][mid] = 1.0
    return grid


def generate_x(n: int) -> Matrix:
    """n×n X 패턴을 자동 생성한다 (보너스: 패턴 생성기)."""
    grid = [[0.0] * n for _ in range(n)]
    for i in range(n):
        grid[i][i] = 1.0
        grid[i][n - 1 - i] = 1.0
    return grid
