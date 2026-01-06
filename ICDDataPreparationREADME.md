The example ICD data file is constructed in the following manner:

1. Download "Code Descriptions in Tabular Order" from CMS.gov ICD-10 [page](https://www.cms.gov/medicare/coding-billing/icd-10-codes) and similarly "DESC_LONG_DX" (long diagnosis descriptions) from the CMS.gov ICD9 [page](https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-9-cm-diagnosis-procedure-codes-abbreviated-and-full-code-titles).
2.  Read ICD-10 codes from 2021-2026 and ICD-9 codes from 2010-2014, then merges each coding system's data across all available years. Identify when each code first appeared and last appeared, and captures the most recent description for each code.
3. Added proper decimal formatting (e.g., E8061 → E806.1) according to following rules, and then combined ICD-9 and ICD-10 into a single reference file with standardized columns :
   *  **ICD-9-CM** codes are 3-6 characters in length with the first character being numeric or alpha (E or V) and rest of the characters being numeric. Use of decimal after 3 charcaters except E codes where the decimal is between third and fourth digit.
   *  **ICD-10-CM** codes are 3 -7 characters in lenght with the first character being alpha (all letters except U are used), character 2 being numeric and rest are alpha or numeric. Use of decimal after 3 characters.
  
### *Note* : *The pipeline works best when you are using the actual ICD codes and descriptions that exist within the biobank/EHR data set*
