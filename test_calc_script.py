import sys
from collections import defaultdict


def calculate_accuracy_by_case(filename):
    # 각 케이스별로 전체 개수와 정답 개수를 저장할 딕셔너리
    case_counts = defaultdict(lambda: {'total': 0, 'correct': 0})
    total_samples = 0
    total_correct = 0

    # idx 중복 체크를 위한 set
    seen_indices = set()

    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # 각 라인에서 idx, 케이스, 정확도 추출
                parts = line.strip().split(', ')
                idx = int(parts[0].split(': ')[1])
                case = int(parts[1].split(': ')[1])
                acc = float(parts[2].split(': ')[1])

                # idx 중복 체크
                assert idx not in seen_indices, f"중복된 idx 발견: {idx} (라인 {line_num})"
                seen_indices.add(idx)

                # 전체 카운트 증가
                total_samples += 1
                if acc == 1.0:
                    total_correct += 1

                # 케이스별 카운트 증가
                case_counts[case]['total'] += 1
                if acc == 1.0:
                    case_counts[case]['correct'] += 1

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except AssertionError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

    # 결과 출력
    print("\n결과:")
    print("=" * 50)
    print("전체 정확도:")
    total_ratio = total_correct / total_samples if total_samples > 0 else 0
    print(f"전체: {total_correct}/{total_samples} = {total_ratio:.4f}")
    print("=" * 50)

    print("\n케이스별 정확도:")
    print("-" * 50)
    for case in sorted(case_counts.keys()):
        total = case_counts[case]['total']
        correct = case_counts[case]['correct']
        ratio = correct / total if total > 0 else 0
        print(f"Case {case}:")
        print(f"정확도: {correct}/{total} = {ratio:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)

    filename = sys.argv[1]
    calculate_accuracy_by_case(filename)