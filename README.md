# Task 7 - Baseline Target Feature Engineering

**PlaceMux · Altrodav Technologies Pvt. Ltd. · Phase 1 Industry Immersion**

## Objective
Engineer the baseline target and core features: build the signals that turn raw data into learnable structure.

## Dataset
Campus Recruitment dataset with 215 students and 15 columns.
- Target: status (Placed / Not Placed) - 148 Placed, 67 Not Placed
- Leaky column salary dropped

## Features Engineered
- score_trend_ssc_hsc: Academic improvement from 10th to 12th
- avg_academic_score: Average of SSC, HSC, and Degree scores
- all_above_60: Flag if all academic scores above 60
- mba_with_workex: Strong MBA score and work experience combo

## Results
- Total features engineered: 18
- Features selected after pruning: 11
- Validation accuracy: 84.4%
- Top feature: ssc_p with 18.0% importance

## Leakage Check
- No single feature exceeded 40% importance threshold
- Train/Val gap due to small dataset size, not leakage

## Tools Used
- pandas, scikit-learn, matplotlib, seaborn
- Random Forest for feature importance ranking

## How to Run
python task7_features.py
