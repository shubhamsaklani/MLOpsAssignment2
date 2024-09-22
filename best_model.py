# regression_model_pipeline.py

# Import necessary libraries
from pycaret.datasets import get_data
from pycaret.regression import *

def main():
    # Load dataset
    print("Loading dataset...")
    data = get_data('insurance')
    print("Dataset loaded.")

    # Setup PyCaret environment
    print("Setting up PyCaret environment...")
    s = setup(data, target='charges', session_id=123)
    print("PyCaret environment set up.")

    # Compare models and select the best one
    print("Comparing models...")
    best = compare_models(include=['rf', 'catboost', 'dt', 'lightgbm', 'et'])
    print(f"Best model: {best}")

    # Tune the best model
    print("Tuning the best model...")
    tuned_model = tune_model(best)
    print("Model tuning complete.")

    # Finalize the best model
    print("Finalizing the model...")
    best_model_finalize = finalize_model(tuned_model)
    print(f"Final model: {best_model_finalize}")

    # Save the model
    print("Saving the model...")
    save_model(best_model_finalize, 'best_model')
    print("Model saved as 'best_model.pkl'.")

    # Evaluate the final model
    print("Evaluating the model...")
    evaluate_model(best_model_finalize)
    print("Model evaluation complete.")

    # Plotting residuals
    print("Plotting residuals...")
    plot_model(best_model_finalize, plot='residuals')
    
    # Plotting feature importance
    print("Plotting feature importance...")
    plot_model(best_model_finalize, plot='feature')

    # Plotting prediction error
    print("Plotting prediction error...")
    plot_model(best_model_finalize, plot='error')

    # Predict using the model
    print("Predicting with the finalized model...")
    predict_model(best_model_finalize)

    # Model interpretation
    print("Interpreting the model...")
    interpret_model(best)
    print("Model interpretation complete.")

if __name__ == "__main__":
    main()
