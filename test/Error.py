import os
import re
import pandas as pd
import numpy as np

def find_jizai_file(ref_file_path: str, search_folder: str) -> str:
    """
    根据第二个文件（PR文件）的文件名，在指定文件夹中找到对应的JiZai文件。
    匹配规则：文件名中同时包含 key_segment、date_segment 和 "JiZai"
    """
    ref_name = os.path.basename(ref_file_path)

    # 从PR文件名中提取关键段，例如 PW_FDC_C_1d5_H0800_M120 和 20260130
    # 文件名格式：RESULT_PR_<key_segment>_<extra>_<date>_<version>.dat
    # 用正则提取：PR_之后到_XGD（或其他后缀）之前为key_segment，以及8位日期
    key_match = re.search(r'PR_([\w]+(?:_[\w]+)*)_XGD', ref_name)
    date_match = re.search(r'_(\d{8})_', ref_name)

    if not key_match or not date_match:
        raise ValueError(f"无法从文件名中提取关键信息：{ref_name}")

    key_segment = key_match.group(1)   # e.g. PW_FDC_C_1d5_H0800_M120
    date_segment = date_match.group(1) # e.g. 20260130

    print(f"提取的关键段：{key_segment}")
    print(f"提取的日期段：{date_segment}")

    # 在 search_folder 中查找包含这三个字段的 .dat 文件
    for fname in os.listdir(search_folder):
        if not fname.endswith('.dat'):
            continue
        if key_segment in fname and date_segment in fname and 'JiZai' in fname:
            return os.path.join(search_folder, fname)

    raise FileNotFoundError(
        f"在文件夹 '{search_folder}' 中未找到包含 '{key_segment}'、'{date_segment}' 和 'JiZai' 的 .dat 文件"
    )


def calc_relative_error(ref_file_path: str, search_folder: str, output_path: str = None):
    """
    主函数：
    1. 根据ref_file_path（PR文件）在search_folder中找到对应的JiZai文件
    2. 读取两个文件
    3. 按通道名匹配行，计算后9列的相对误差
    4. 输出结果
    """
    # --- 1. 找到 JiZai 文件 ---
    jizai_path = find_jizai_file(ref_file_path, search_folder)
    print(f"找到对应文件：{jizai_path}")

    # --- 2. 读取文件 ---
    df_pr = pd.read_csv(ref_file_path)
    df_jz = pd.read_csv(jizai_path)

    # PR文件列：ch, freq1, damp1, pp1, freq2, damp2, pp2, freq3, damp3, pp3
    # JiZai文件列：有效性,H,M,激励,相位,响应,wn_1,zeta_1,t_val_1,wn_2,zeta_2,t_val_2,wn_3,zeta_3,t_val_3
    pr_ch_col = 'ch'
    jz_ch_col = '响应'

    pr_val_cols = ['freq1', 'damp1', 'pp1', 'freq2', 'damp2', 'pp2', 'freq3', 'damp3', 'pp3']
    jz_val_cols = ['wn_1', 'zeta_1', 't_val_1', 'wn_2', 'zeta_2', 't_val_2', 'wn_3', 'zeta_3', 't_val_3']

    # 将JiZai文件以响应列为索引，方便查找
    df_jz_indexed = df_jz.set_index(jz_ch_col)

    # --- 3. 逐通道匹配并计算相对误差 ---
    result_rows = []
    unmatched = []

    for _, pr_row in df_pr.iterrows():
        ch = pr_row[pr_ch_col]

        if ch not in df_jz_indexed.index:
            unmatched.append(ch)
            # 通道在JiZai文件中不存在，填NaN
            result_rows.append([ch] + [np.nan] * 9)
            continue

        jz_row = df_jz_indexed.loc[ch]

        err_vals = []
        for pr_col, jz_col in zip(pr_val_cols, jz_val_cols):
            pr_val = pr_row[pr_col]
            jz_val = jz_row[jz_col]

            # 转换为float，空字符串/缺失值视为NaN
            try:
                pr_val = float(pr_val)
            except (ValueError, TypeError):
                pr_val = np.nan
            try:
                jz_val = float(jz_val)
            except (ValueError, TypeError):
                jz_val = np.nan

            # 相对误差 = |pr - jz| / |jz| * 100
            if pd.isna(pr_val) or pd.isna(jz_val):
                err_vals.append(np.nan)
            elif jz_val == 0:
                # 分母为0时无法计算相对误差
                err_vals.append(np.nan)
            else:
                err_vals.append(abs(pr_val - jz_val) / abs(jz_val) * 100)

        result_rows.append([ch] + err_vals)

    if unmatched:
        print(f"警告：以下通道在JiZai文件中未找到匹配：{unmatched}")

    # --- 4. 构建结果DataFrame ---
    out_cols = ['ch', 'freq_err1', 'damp_err1', 'pp_err1',
                'freq_err2', 'damp_err2', 'pp_err2',
                'freq_err3', 'damp_err3', 'pp_err3']
    df_result = pd.DataFrame(result_rows, columns=out_cols)

    # --- 5. 输出 ---
    if output_path is None:
        ref_dir = os.path.dirname(ref_file_path)
        ref_name = os.path.basename(ref_file_path)
        out_name = ref_name.replace('.dat', '_ERROR.dat')
        output_path = os.path.join(ref_dir, out_name)

    df_result.to_csv(output_path, index=False)
    print(f"结果已保存至：{output_path}")
    return df_result


# ============================================================
# 使用示例（直接运行此脚本时执行）
# ============================================================
if __name__ == '__main__':
    # ---- 请修改以下两个路径 ----
    REF_FILE   = r"RESULT_PR_PW_FDC_C_1d5_H0800_M120_XGD_mx_20260130_v1.dat"  # 第二个文件（PR文件）路径
    SEARCH_DIR = r"."   # 存放JiZai文件的文件夹路径
    OUTPUT     = None   # 输出路径，None表示自动生成（与PR文件同目录，文件名加_ERROR后缀）
    # ----------------------------

    df = calc_relative_error(REF_FILE, SEARCH_DIR, OUTPUT)
    print("\n结果预览：")
    print(df.to_string())