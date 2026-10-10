"""Command-line entry point for the no-code OT repository import wizard."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ot_benchmark"))

from algorithms.plugin_importer import run_import_wizard


def main():
    parser = argparse.ArgumentParser(description="无代码导入第三方 OT 算法仓库")
    parser.add_argument("repository", help="下载后的 GitHub 仓库目录")
    parser.add_argument(
        "--auto", action="store_true",
        help="自动选择评分最高且不需要额外参数的函数",
    )
    args = parser.parse_args()
    run_import_wizard(args.repository, auto=args.auto)


if __name__ == "__main__":
    main()
