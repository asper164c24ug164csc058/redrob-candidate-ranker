#!/usr/bin/env python3
"""Convert the rank.py CSV output to XLSX (the H2S/Redrob portal upload
form asks for the ranked output as .xlsx, even though submission_spec.docx
describes a .csv). Keep both on hand."""

import argparse
import csv
from openpyxl import Workbook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wb = Workbook()
    ws = wb.active
    ws.title = "top_100_ranking"
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            ws.append(row)

    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 8
    ws.column_dimensions["C"].width = 10
    ws.column_dimensions["D"].width = 100
    wb.save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
