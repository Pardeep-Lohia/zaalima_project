from src.complete_pipeline import run_complete_pipeline, print_pipeline_summary
results = run_complete_pipeline(regenerate_data=True, model_type='xgboost', tune_hyperparams=True, n_trials=20, save_all=True)
print_pipeline_summary(results)
