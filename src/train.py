import mlflow
import mlflow.sklearn
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import pandas as pd
import pickle

def run_ml_ops_logging():
    mlflow.set_experiment("CINEIQ_Production_Core")
    
    with mlflow.start_run():
        print("-> Ingesting metrics context tracking lines into MLflow...")
        df = pd.read_csv("CineIq_Data/demo_ratings.csv")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
        
        n_factors = 100
        epochs = 20
        lr_all = 0.005
        
        mlflow.log_param("matrix_factors", n_factors)
        mlflow.log_param("optimization_epochs", epochs)
        mlflow.log_param("learning_rate", lr_all)
        
        algo = SVD(n_factors=n_factors, n_epochs=epochs, lr_all=lr_all, random_state=42)
        cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
        
        mlflow.log_metric("validation_rmse", cv_results['test_rmse'].mean())
        mlflow.log_metric("validation_mae", cv_results['test_mae'].mean())
        
        trainset = data.build_full_trainset()
        algo.fit(trainset)
        
        with open("CineIq_Data/modify/svd_model.pkl", "wb") as f:
            pickle.dump(algo, f)
            
        mlflow.log_artifact("CineIq_Data/modify/svd_model.pkl")
        print("-> MLflow pipeline tracking session complete.")

if __name__ == "__main__":
    run_ml_ops_logging()