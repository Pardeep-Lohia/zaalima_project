from src.complete_pipeline import run_complete_pipeline, print_pipeline_summary

# Test with different model type (random forest) and fewer trials
results = run_complete_pipeline(regenerate_data=False, model_type='random_forest', tune_hyperparams=True, n_trials=5, save_all=False)
print_pipeline_summary(results)

# Test with baseline model
results_baseline = run_complete_pipeline(regenerate_data=False, model_type='baseline', tune_hyperparams=False, n_trials=0, save_all=False)
print_pipeline_summary(results_baseline)
