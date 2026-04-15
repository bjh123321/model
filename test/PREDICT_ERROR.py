from pathlib import Path
from Error import calc_relative_error


def calc_error_for_all(result_pr_dir: Path, result_cg_dir: Path, error_out_dir: Path):
    """
    批量计算误差：
    - 输入：RESULT_PR_XGD 文件夹
    - 对比：RESULT_CG 文件夹
    - 输出：ERROR_PR_XGD 文件夹
    """

    error_out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(result_pr_dir.glob("RESULT_PR_*.dat"))

    if len(files) == 0:
        print("没有找到 RESULT_PR 文件")
        return

    print(f"找到 {len(files)} 个文件，开始计算误差...")

    for fd_path in files:
        # 生成对应 error 文件名
        error_name = fd_path.name.replace("RESULT_PR_", "ERROR_PR_")
        error_path = error_out_dir / error_name

        try:
            calc_relative_error(fd_path, result_cg_dir, error_path)
            print(f"[OK] {fd_path.name}")
        except Exception as e:
            print(f"[ERROR] {fd_path.name} -> {e}")


if __name__ == "__main__":
    1.误差exe放在哪个文件夹
    2.算哪些状态点的误差？任务单的误差？
    result_pr_dir = Path(r"D:\Projects\GeneralParameter\J35_001_none\PR\XGD\XGD_mx_20260130_v1\RESULT_PR_XGD")
    result_cg_dir = Path(r"D:\Projects\GeneralParameter\J35_001_none\RESULT_CG")
    error_out_dir = Path(r"D:\Projects\GeneralParameter\J35_001_none\PR\XGD\XGD_mx_20260130_v1\ERROR_PR_XGD")

    calc_error_for_all(result_pr_dir, result_cg_dir, error_out_dir)