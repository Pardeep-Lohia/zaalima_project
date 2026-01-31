# Predictive Maintenance Pipeline - Pending Tasks

## 1. Threshold Optimization ✅ (Already Implemented)
- [x] Fixed recall targets (70%, 80%, 90%)
- [x] Fixed alert budget (≤ N false positives per day)
- [x] Threshold vs precision/recall table
- [x] Updated confusion matrices at optimized thresholds
- [x] Do NOT use default 0.5

## 2. Imbalance-Aware Model Upgrade ✅ (Already Implemented)
- [x] XGBoost with scale_pos_weight
- [x] LightGBM with scale_pos_weight
- [x] Time-aware train/validation split (no shuffling)
- [x] Use PR-AUC as primary metric
- [x] Compare against existing baselines fairly

## 3. Synthetic Failure Signal Injection ✅ (Completed)
- [x] Improve synthetic data generator for predictable failures
- [x] Sustained high vibration precursors
- [x] Rising rolling variance
- [x] Temperature drift
- [x] Pressure instability 12-48 hours before failure
- [x] Ensure no label leakage
- [x] Signal appears gradually, not instantaneously
- [x] Re-train and re-evaluate models after injection

## 4. Proper Evaluation for Rare Events ✅ (Completed)
- [x] Create enhanced_evaluation.py module
- [x] Recall @ fixed false-positive rate
- [x] Precision @ top-K alerts
- [x] Daily alert volume simulation
- [x] Improve PR-AUC interpretation
- [x] Avoid accuracy-only reporting
- [x] Avoid ROC-AUC as primary metric

## 5. Clear Deliverables 🔄 (In Progress)
- [ ] Update training/evaluation code
- [ ] Save plots to /reports
- [ ] Create markdown/text summary explaining:
  - Why metrics improved or didn't
  - Trade-offs between recall and alert volume
  - Which model + threshold recommended and why
- [ ] Ensure interpretable by non-ML stakeholder

## Success Criteria
- [ ] Model produces meaningfully better than random PR-AUC
- [ ] Failure recall can be increased without exploding false positives
- [ ] Threshold selection explicit and justified
- [ ] Results interpretable by non-ML stakeholder
