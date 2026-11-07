
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POWERBI_DIR = PROJECT_ROOT / "powerbi"

sheets = {
    "Model Performance": "model_performance.csv",
    "CV Summary": "cv_summary.csv",
    "Dataset Overview": "dataset_overview.csv",
    "Class Distribution": "class_distribution.csv",
    "Error Summary": "error_summary.csv",
    "Error Details": "error_details.csv",
}

output_path = POWERBI_DIR / "JobShield_AI_PowerBI_Data.xlsx"

# Write each dataset to a separate worksheet
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for sheet_name, filename in sheets.items():
        df = pd.read_csv(POWERBI_DIR / filename)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# Format each worksheet as an Excel table
workbook = load_workbook(output_path)

for index, worksheet in enumerate(workbook.worksheets, start=1):
    if worksheet.max_row < 2:
        continue

    table_name = f"JobShieldTable{index}"

    table = Table(
        displayName=table_name,
        ref=worksheet.dimensions
    )

    style = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )

    table.tableStyleInfo = style
    worksheet.add_table(table)

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    for column in worksheet.columns:
        max_length = max(
            len(str(cell.value or ""))
            for cell in column
        )
        worksheet.column_dimensions[
            column[0].column_letter
        ].width = min(max_length + 3, 45)

workbook.save(output_path)

# Verify workbook
check = load_workbook(output_path, read_only=False)

print("Power BI workbook created successfully.")
print(f"File: {output_path}")
print("\nWorksheets:")

for worksheet in check.worksheets:
    print(
        f"- {worksheet.title}: "
        f"{worksheet.max_row - 1} data rows"
    )

print(f"\nFile size: {output_path.stat().st_size / 1024:.1f} KB")
