"""Main training pipeline for chemical reaction rate prediction."""

import argparse
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.data.data_generator import ArrheniusDataGenerator
from src.data.data_loader import ReactionDataLoader
from src.models.traditional_models import (
    LinearRegressionModel,
    PolynomialRegressionModel,
    SVRModel,
    RandomForestModel,
)
from src.evaluation.metrics import RegressionMetrics
from sklearn.model_selection import cross_val_score, KFold

console = Console()


def generate_data(output_path: Path, num_samples: int = 300):
    """Generate synthetic reaction data."""
    console.print("[bold blue]Generating synthetic data...[/bold blue]")
    generator = ArrheniusDataGenerator()
    data = generator.generate_data(num_samples=num_samples)
    generator.save_data(data, output_path)
    console.print(f"[green]✓ Data saved to {output_path}[/green]")
    return data


def train_and_evaluate(data_path: Path):
    """Train multiple models and evaluate performance."""
    # Load data
    console.print("\n[bold blue]Loading data...[/bold blue]")
    loader = ReactionDataLoader(data_path)
    loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data()

    # Define models to compare
    models = {
        "Linear Regression": LinearRegressionModel(),
        "Polynomial Regression": PolynomialRegressionModel({"degree": 2}),
        "SVR": SVRModel(),
        "Random Forest": RandomForestModel(),
    }

    # Cross-validation
    console.print("\n[bold blue]Running cross-validation...[/bold blue]")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(
            model.model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1
        )
        cv_results[name] = scores
        console.print(
            f"  {name}: R² = {scores.mean():.4f} (±{scores.std():.4f})"
        )

    # Train and evaluate all models
    console.print("\n[bold blue]Training and evaluating models...[/bold blue]")
    test_results = {}

    for name, model in models.items():
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = RegressionMetrics.calculate_metrics(y_test, y_pred)
        test_results[name] = metrics

    # Display results in a table
    table = Table(title="Model Performance Comparison")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("MAE", justify="right", style="green")
    table.add_column("RMSE", justify="right", style="green")
    table.add_column("R²", justify="right", style="green")

    for name, metrics in test_results.items():
        table.add_row(
            name,
            f"{metrics['MAE']:.6f}",
            f"{metrics['RMSE']:.6f}",
            f"{metrics['R2']:.6f}",
        )

    console.print("\n")
    console.print(table)

    # Find best model
    best_model_name = max(test_results, key=lambda k: test_results[k]["R2"])
    console.print(f"\n[bold green]Best model: {best_model_name}[/bold green]")

    # Make prediction on new data
    best_model = models[best_model_name]
    new_condition = pd.DataFrame(
        [[80, 1.5, 1]], columns=["temperature_C", "concentration_mol_L", "catalyst"]
    )
    prediction = best_model.predict(new_condition)

    console.print("\n[bold blue]Prediction on new condition:[/bold blue]")
    console.print(f"  Temperature: 80°C")
    console.print(f"  Concentration: 1.5 mol/L")
    console.print(f"  Catalyst: Yes")
    console.print(f"  [bold green]Predicted rate: {prediction[0]:.4f} mol/L·s[/bold green]")

    return best_model, test_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Chemical Reaction Rate Prediction ML Pipeline"
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate new synthetic data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=300,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/chemical_reaction_data.csv",
        help="Path to data file",
    )
    args = parser.parse_args()

    console.print("[bold magenta]Chemical Reaction Rate Prediction ML[/bold magenta]")
    console.print("[magenta]Modern Machine Learning Framework[/magenta]\n")

    data_path = Path(args.data_path)

    # Generate data if requested or if file doesn't exist
    if args.generate_data or not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        generate_data(data_path, args.num_samples)

    # Train and evaluate
    train_and_evaluate(data_path)

    console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")


if __name__ == "__main__":
    main()
