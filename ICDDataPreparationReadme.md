## Merged ICD File Creation

The merged ICD file combines ICD-9 and ICD-10 codes from multiple years into a single reference dataset.

### Process:

1. **Load ICD-10 codes** from 2021-2026 CMS files [https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-9-cm-diagnosis-procedure-codes-abbreviated-and-full-code-titles]
2. **Load ICD-9 codes** from 2010-2014 CMS files [Link]
4. **Extract temporal information**:
   - First appearance year (earliest year code exists)
   - Last appearance year (most recent year code exists)
   - Most updated description (latest available description)

5. **Add formatting**: [Write better]
   - ICD-9 E-codes: `E8061` → `E806.1` (E###.#)
   - Other codes: `25000` → `250.00` (###.##)

6. **Combine ICD-9 and ICD-10**:
   - Outer merge to include all codes from both systems
   - Flag codes appearing in both systems
   - Consolidate temporal and description fields

### Output columns:
- `code`: Original unformatted code
- `icd_code`: Formatted code with decimal
- `First_appearance`: Year code first appeared
- `Last_appearance`: Year code last appeared  
- `icd_string`: Most recent description
- `Code_system`: ICD9 or ICD10

This provides a comprehensive historical view of all ICD codes across both coding systems.
