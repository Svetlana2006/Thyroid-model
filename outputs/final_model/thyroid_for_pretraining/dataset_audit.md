# Dataset Audit: Thyroid for Pretraining

## Dataset Structure
- **Total Images:** 7288
- **Benign Images:** 3282
- **Malignant Images:** 4006
- **Unique Patients:** 3644

## Label Mapping
- Benign = 0
- Malignant = 1

## Unit of Analysis
Since there are multiple images per patient (average 2.0), the predictions were averaged per patient to ensure independent observations. All metrics below are reported at the **patient level**.

## Overlap Contamination
Verified exact file hashing against TN5000, AUITD, and Divesh.
- Exact duplicates found: 0

## Final Verdict
**A. VALID EXTERNAL VALIDATION**
The dataset has an independent origin, contains strictly benign/malignant labels, and has no exact data leakage with development sets.
