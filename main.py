"""Mini NPU Simulator - 메인 실행 파일

MAC(Multiply-Accumulate) 연산을 통해 패턴을 판별하는 시뮬레이터.
모드 1: 사용자가 3×3 필터/패턴을 직접 입력
모드 2: data.json에서 다양한 크기의 필터/패턴을 로드하여 일괄 분석
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Tuple

from npu_core import (
    BENCHMARK_REPEATS,
    EPSILON,
    LABEL_CROSS,
    LABEL_UNDECIDED,
    LABEL_X,
    Matrix,
    benchmark_mac,
    generate_cross,
    generate_x,
    judge,
    mac,
    mac_flat,
    normalize_label,
    validate_matrix,
)

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")


# ── 입력 유틸리티 ──────────────────────────────────────


def read_matrix(prompt: str, n: int) -> Matrix:
    """n줄을 입력받아 n×n 행렬로 반환한다. 오류 시 재입력을 유도한다."""
    while True:
        print(prompt)
        rows: Matrix = []
        valid = True
        for line_no in range(1, n + 1):
            raw = input().strip()
            parts = raw.split()
            if len(parts) != n:
                print(f"  입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요.")
                valid = False
                break
            try:
                row = [float(x) for x in parts]
            except ValueError:
                print(f"  입력 형식 오류: 숫자가 아닌 값이 포함되어 있습니다.")
                valid = False
                break
            rows.append(row)
        if valid and len(rows) == n:
            return rows
        print(f"  다시 입력하세요.\n")


def print_matrix(matrix: Matrix, indent: str = "  ") -> None:
    """행렬을 보기 좋게 출력한다."""
    for row in matrix:
        print(indent + "  ".join(f"{v:g}" for v in row))


# ── 모드 1: 사용자 입력 (3×3) ──────────────────────────


def mode_user_input() -> None:
    """3×3 필터 2개와 패턴을 입력받아 MAC 연산 및 판정을 수행한다."""
    n = 3

    print()
    print("#" + "-" * 40)
    print("# [1] 필터 입력")
    print("#" + "-" * 40)

    filter_a = read_matrix(f"필터 A ({n}줄 입력, 공백 구분)", n)
    print()
    filter_b = read_matrix(f"필터 B ({n}줄 입력, 공백 구분)", n)

    print()
    print("#" + "-" * 40)
    print("# [2] 패턴 입력")
    print("#" + "-" * 40)

    pattern = read_matrix(f"패턴 ({n}줄 입력, 공백 구분)", n)

    # MAC 연산
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)

    # 성능 측정
    avg_ms_a = benchmark_mac(pattern, filter_a)
    avg_ms_b = benchmark_mac(pattern, filter_b)
    avg_ms = (avg_ms_a + avg_ms_b) / 2

    # 판정
    diff = abs(score_a - score_b)
    if diff < EPSILON:
        verdict = "판정 불가"
    elif score_a > score_b:
        verdict = "A"
    else:
        verdict = "B"

    print()
    print("#" + "-" * 40)
    print("# [3] MAC 결과")
    print("#" + "-" * 40)
    print(f"  A 점수: {score_a}")
    print(f"  B 점수: {score_b}")
    print(f"  연산 시간(평균/{BENCHMARK_REPEATS}회): {avg_ms:.3f} ms")

    if diff < EPSILON:
        print(f"  판정: 판정 불가 (|A-B| < {EPSILON})")
    else:
        print(f"  판정: {verdict}")

    # 성능 분석
    print()
    print("#" + "-" * 40)
    print(f"# [4] 성능 분석 (평균/{BENCHMARK_REPEATS}회)")
    print("#" + "-" * 40)
    print(f"  {'크기':<10} {'평균 시간(ms)':<16} {'연산 횟수'}")
    print("  " + "-" * 38)
    print(f"  {n}x{n:<8} {avg_ms:<16.3f} {n*n}")

    # 보너스: 1차원 최적화 비교
    print()
    print("#" + "-" * 40)
    print("# [보너스] 2D vs 1D 최적화 비교")
    print("#" + "-" * 40)
    import time

    start = time.perf_counter()
    for _ in range(BENCHMARK_REPEATS):
        mac(pattern, filter_a)
    t_2d = (time.perf_counter() - start) / BENCHMARK_REPEATS * 1000

    start = time.perf_counter()
    for _ in range(BENCHMARK_REPEATS):
        mac_flat(pattern, filter_a)
    t_1d = (time.perf_counter() - start) / BENCHMARK_REPEATS * 1000

    print(f"  2D MAC: {t_2d:.4f} ms")
    print(f"  1D MAC: {t_1d:.4f} ms")


# ── 모드 2: data.json 분석 ─────────────────────────────


def load_json() -> Optional[dict]:
    """data.json을 로드한다."""
    if not os.path.exists(DATA_FILE):
        print(f"  오류: {DATA_FILE} 파일을 찾을 수 없습니다.")
        return None
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  오류: data.json 파싱 실패 - {e}")
        return None


def extract_size_from_key(key: str) -> Optional[int]:
    """패턴 키(예: 'size_5_1')에서 크기 N을 추출한다."""
    parts = key.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def mode_json_analysis() -> None:
    """data.json에서 필터/패턴을 로드하여 일괄 판정한다."""
    data = load_json()
    if data is None:
        return

    filters_raw = data.get("filters", {})
    patterns_raw = data.get("patterns", {})

    # [1] 필터 로드
    print()
    print("#" + "-" * 40)
    print("# [1] 필터 로드")
    print("#" + "-" * 40)

    filters: dict[int, dict[str, Matrix]] = {}

    for size_key, filter_dict in sorted(filters_raw.items()):
        parts = size_key.split("_")
        if len(parts) < 2:
            print(f"  ! {size_key}: 키 형식 오류 (size_N 형태가 아님)")
            continue
        try:
            n = int(parts[1])
        except ValueError:
            print(f"  ! {size_key}: 크기 파싱 실패")
            continue

        loaded = {}
        for label_raw, matrix in filter_dict.items():
            label = normalize_label(label_raw)
            ok, msg = validate_matrix(matrix, n)
            if not ok:
                print(f"  ! {size_key}/{label_raw}: {msg}")
                continue
            loaded[label] = matrix

        if LABEL_CROSS in loaded and LABEL_X in loaded:
            filters[n] = loaded
            print(f"  + size_{n:<3} 필터 로드 완료 (Cross, X)")
        else:
            print(f"  ! size_{n}: Cross 또는 X 필터 누락")

    # [2] 패턴 분석
    print()
    print("#" + "-" * 40)
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#" + "-" * 40)

    results: List[Tuple[str, str, bool, str]] = []  # (key, verdict, passed, reason)
    sizes_seen: set[int] = set()

    for pat_key in sorted(patterns_raw.keys(), key=_pattern_sort_key):
        pat_data = patterns_raw[pat_key]
        n = extract_size_from_key(pat_key)

        print(f"\n  --- {pat_key} ---")

        if n is None:
            reason = "키에서 크기 추출 실패"
            print(f"  FAIL: {reason}")
            results.append((pat_key, "ERROR", False, reason))
            continue

        sizes_seen.add(n)

        if n not in filters:
            reason = f"size_{n} 필터 없음"
            print(f"  FAIL: {reason}")
            results.append((pat_key, "ERROR", False, reason))
            continue

        pattern = pat_data.get("input")
        expected_raw = pat_data.get("expected", "")

        if pattern is None:
            reason = "input 필드 누락"
            print(f"  FAIL: {reason}")
            results.append((pat_key, "ERROR", False, reason))
            continue

        ok, msg = validate_matrix(pattern, n)
        if not ok:
            reason = f"패턴 크기 불일치 - {msg}"
            print(f"  FAIL: {reason}")
            results.append((pat_key, "ERROR", False, reason))
            continue

        expected = normalize_label(expected_raw)

        score_cross = mac(pattern, filters[n][LABEL_CROSS])
        score_x = mac(pattern, filters[n][LABEL_X])
        verdict = judge(score_cross, score_x)

        passed = (verdict == expected)
        reason = ""
        if not passed:
            if verdict == LABEL_UNDECIDED:
                reason = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
            else:
                reason = f"판정({verdict}) != expected({expected})"

        status = "PASS" if passed else "FAIL"
        print(f"  Cross 점수: {score_cross}")
        print(f"  X 점수:     {score_x}")
        fail_info = f" ({reason})" if reason else ""
        print(f"  판정: {verdict} | expected: {expected} | {status}{fail_info}")

        results.append((pat_key, verdict, passed, reason))

    # [3] 성능 분석
    print()
    print("#" + "-" * 40)
    print(f"# [3] 성능 분석 (평균/{BENCHMARK_REPEATS}회)")
    print("#" + "-" * 40)

    all_sizes = sorted({3} | sizes_seen)
    print(f"  {'크기':<10} {'평균 시간(ms)':<16} {'연산 횟수'}")
    print("  " + "-" * 38)

    for n in all_sizes:
        p = generate_cross(n)
        f = generate_x(n)
        avg = benchmark_mac(p, f)
        print(f"  {n}x{n:<8} {avg:<16.3f} {n*n}")

    # 보너스: 2D vs 1D 비교
    print()
    print("#" + "-" * 40)
    print("# [보너스] 2D vs 1D 최적화 비교")
    print("#" + "-" * 40)
    print(f"  {'크기':<10} {'2D(ms)':<12} {'1D(ms)':<12}")
    print("  " + "-" * 34)
    import time as _time

    for n in all_sizes:
        p = generate_cross(n)
        f = generate_x(n)

        start = _time.perf_counter()
        for _ in range(BENCHMARK_REPEATS):
            mac(p, f)
        t_2d = (_time.perf_counter() - start) / BENCHMARK_REPEATS * 1000

        start = _time.perf_counter()
        for _ in range(BENCHMARK_REPEATS):
            mac_flat(p, f)
        t_1d = (_time.perf_counter() - start) / BENCHMARK_REPEATS * 1000

        print(f"  {n}x{n:<8} {t_2d:<12.4f} {t_1d:<12.4f}")

    # [4] 결과 요약
    total = len(results)
    passed_count = sum(1 for _, _, p, _ in results if p)
    failed_count = total - passed_count

    print()
    print("#" + "-" * 40)
    print("# [4] 결과 요약")
    print("#" + "-" * 40)
    print(f"  총 테스트: {total}개")
    print(f"  통과: {passed_count}개")
    print(f"  실패: {failed_count}개")

    if failed_count > 0:
        print()
        print("  실패 케이스:")
        for key, verdict, passed, reason in results:
            if not passed:
                print(f"  - {key}: {reason}")

    print()
    print("  (상세 원인 분석 및 복잡도 설명은 README.md의 \"결과 리포트\" 섹션 참고)")


def _pattern_sort_key(key: str) -> Tuple[int, int]:
    """패턴 키를 (크기, 인덱스) 기준으로 정렬한다."""
    parts = key.split("_")
    try:
        return (int(parts[1]), int(parts[2]))
    except (IndexError, ValueError):
        return (0, 0)


# ── 메인 ───────────────────────────────────────────────


def main() -> None:
    print()
    print("=== Mini NPU Simulator ===")
    print()
    print("[모드 선택]")
    print("  1. 사용자 입력 (3x3)")
    print("  2. data.json 분석")

    while True:
        raw = input("선택: ").strip()
        if raw == "1":
            mode_user_input()
            break
        elif raw == "2":
            mode_json_analysis()
            break
        else:
            print("  1 또는 2를 입력하세요.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n\n프로그램이 중단되었습니다.")
        sys.exit(0)
