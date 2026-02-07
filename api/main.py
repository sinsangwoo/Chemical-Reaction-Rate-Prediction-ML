"""FastAPI main application.

REST API for Chemical Reaction Rate Prediction ML Platform.

Endpoints:
- POST /predict: Single reaction prediction
- POST /predict/batch: Batch predictions
- POST /validate/smiles: SMILES validation
- GET /health: Health check
- GET /models: List available models
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path
import time
import traceback
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    MoleculeInput,
    MoleculeValidationResponse,
    HealthResponse,
    ErrorResponse,
    ModelType,
    UncertaintyMethod,
    UncertaintyEstimate
)

from src.data.smiles_parser import SMILESParser, ReactionSMILES
from src.features.molecular_features import MolecularFeatureExtractor

# Initialize FastAPI app
app = FastAPI(
    title="Chemical Reaction ML API",
    description="REST API for molecular property and reaction rate prediction with uncertainty quantification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (for web apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
START_TIME = time.time()
MODEL_CACHE: Dict = {}
SMILES_PARSER = SMILESParser()
REACTION_PARSER = ReactionSMILES()
FEATURE_EXTRACTOR = MolecularFeatureExtractor()


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=type(exc).__name__,
            message=str(exc),
            details={"traceback": traceback.format_exc()}
        ).dict()
    )


# Helper functions
def load_model(model_type: ModelType):
    """Load model (with caching).
    
    Args:
        model_type: Model type to load
    
    Returns:
        Loaded model instance
    """
    if model_type.value in MODEL_CACHE:
        return MODEL_CACHE[model_type.value]
    
    # Load appropriate model
    if model_type == ModelType.RANDOM_FOREST:
        from src.models.traditional_models import RandomForestModel
        model = RandomForestModel()
        # TODO: Load pre-trained weights
    
    elif model_type in [ModelType.GCN, ModelType.GAT, ModelType.GIN, ModelType.MPNN]:
        try:
            from src.models.gnn.gnn_models import create_gnn_model
            model = create_gnn_model(model_type.value.upper())
            # TODO: Load pre-trained weights
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"GNN models require PyTorch. Model {model_type.value} not available."
            )
    
    elif model_type in [ModelType.MC_DROPOUT, ModelType.BAYESIAN_GNN, ModelType.ENSEMBLE]:
        try:
            from src.models.uncertainty.bayesian_gnn import create_uncertainty_model
            
            type_map = {
                ModelType.MC_DROPOUT: "mc_dropout",
                ModelType.BAYESIAN_GNN: "bayesian",
                ModelType.ENSEMBLE: "ensemble"
            }
            
            model = create_uncertainty_model(type_map[model_type])
            # TODO: Load pre-trained weights
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Bayesian models require PyTorch. Model {model_type.value} not available."
            )
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model type: {model_type.value}"
        )
    
    # Cache model
    MODEL_CACHE[model_type.value] = model
    
    return model


def parse_reaction(request: PredictionRequest) -> Dict:
    """Parse and validate reaction.
    
    Args:
        request: Prediction request
    
    Returns:
        Parsed reaction components
    """
    reaction = request.reaction
    
    # Validate reactants
    for i, smiles in enumerate(reaction.reactants):
        if not SMILES_PARSER.is_valid_smiles(smiles):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid reactant SMILES at index {i}: {smiles}"
            )
    
    # Validate products
    for i, smiles in enumerate(reaction.products):
        if not SMILES_PARSER.is_valid_smiles(smiles):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid product SMILES at index {i}: {smiles}"
            )
    
    # Validate agents
    if reaction.agents:
        for i, smiles in enumerate(reaction.agents):
            if not SMILES_PARSER.is_valid_smiles(smiles):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid agent SMILES at index {i}: {smiles}"
                )
    
    return {
        "reactants": reaction.reactants,
        "products": reaction.products,
        "agents": reaction.agents or [],
        "conditions": reaction.conditions
    }


# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Chemical Reaction ML API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health_check():
    """Health check endpoint.
    
    Returns:
        Health status and system information
    """
    uptime = time.time() - START_TIME
    
    # Check which models are available
    available_models = ["random_forest"]
    
    try:
        import torch
        available_models.extend(["gcn", "gat", "gin", "mpnn"])
        available_models.extend(["mc_dropout", "bayesian_gnn", "ensemble"])
    except ImportError:
        pass
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=available_models,
        uptime_seconds=uptime
    )


@app.get("/models", tags=["Models"])
async def list_models():
    """List available models and their capabilities.
    
    Returns:
        Dictionary of available models
    """
    models = {
        "traditional": {
            "random_forest": {
                "description": "RandomForest regressor",
                "uncertainty": False,
                "speed": "fast"
            }
        },
        "gnn": {},
        "bayesian": {}
    }
    
    try:
        import torch
        models["gnn"] = {
            "gcn": {
                "description": "Graph Convolutional Network",
                "uncertainty": False,
                "speed": "medium"
            },
            "gat": {
                "description": "Graph Attention Network",
                "uncertainty": False,
                "speed": "medium"
            },
            "gin": {
                "description": "Graph Isomorphism Network",
                "uncertainty": False,
                "speed": "medium"
            },
            "mpnn": {
                "description": "Message Passing Neural Network",
                "uncertainty": False,
                "speed": "medium"
            }
        }
        
        models["bayesian"] = {
            "mc_dropout": {
                "description": "Monte Carlo Dropout",
                "uncertainty": True,
                "speed": "medium"
            },
            "bayesian_gnn": {
                "description": "Bayesian GNN with variational inference",
                "uncertainty": True,
                "speed": "slow"
            },
            "ensemble": {
                "description": "Deep Ensemble (5 models)",
                "uncertainty": True,
                "speed": "slow"
            }
        }
    except ImportError:
        models["gnn"] = {"note": "PyTorch not installed"}
        models["bayesian"] = {"note": "PyTorch not installed"}
    
    return models


@app.post("/validate/smiles", response_model=MoleculeValidationResponse, tags=["Validation"])
async def validate_smiles(molecule: MoleculeInput):
    """Validate SMILES string and return molecular properties.
    
    Args:
        molecule: Molecule with SMILES to validate
    
    Returns:
        Validation result with properties
    """
    smiles = molecule.smiles
    
    # Validate
    is_valid = SMILES_PARSER.is_valid_smiles(smiles)
    
    if not is_valid:
        return MoleculeValidationResponse(
            is_valid=False,
            errors=["Invalid SMILES syntax"]
        )
    
    # Extract properties
    try:
        features = SMILES_PARSER.extract_features(smiles)
        
        return MoleculeValidationResponse(
            is_valid=True,
            smiles=smiles,  # Could canonicalize here
            properties={
                "length": features["length"],
                "num_atoms": sum(features["atom_counts"].values()),
                "num_rings": features["num_rings"],
                "has_aromatic": features["has_aromatic"],
                "num_branches": features["num_branches"],
                "estimated_mw": features["estimated_mw"],
                "atom_counts": features["atom_counts"]
            }
        )
    except Exception as e:
        return MoleculeValidationResponse(
            is_valid=False,
            errors=[f"Feature extraction failed: {str(e)}"]
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_reaction_rate(request: PredictionRequest):
    """Predict reaction rate with uncertainty.
    
    Args:
        request: Prediction request with reaction and model config
    
    Returns:
        Prediction with uncertainty estimate
    """
    # Parse and validate reaction
    parsed = parse_reaction(request)
    
    # Load model
    model = load_model(request.model_type)
    
    # TODO: Actual prediction logic
    # For now, return mock prediction
    
    import random
    prediction_value = random.uniform(0.01, 1.0)
    
    # Mock uncertainty if requested
    uncertainty = None
    
    if request.uncertainty_method != UncertaintyMethod.NONE:
        epistemic = random.uniform(0.001, 0.1)
        aleatoric = random.uniform(0.001, 0.05)
        total = epistemic + aleatoric
        
        lower = prediction_value - 1.96 * (total ** 0.5)
        upper = prediction_value + 1.96 * (total ** 0.5)
        
        uncertainty = UncertaintyEstimate(
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total,
            confidence_interval_95=[lower, upper]
        )
    
    return PredictionResponse(
        prediction=prediction_value,
        uncertainty=uncertainty,
        model_used=request.model_type.value,
        metadata={
            "reactants": parsed["reactants"],
            "products": parsed["products"],
            "temperature": parsed["conditions"].temperature,
            "note": "Mock prediction - model not trained yet"
        }
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction for multiple reactions.
    
    Args:
        request: Batch prediction request
    
    Returns:
        Batch prediction results
    """
    predictions = []
    errors = 0
    
    for i, reaction_input in enumerate(request.reactions):
        try:
            # Create individual prediction request
            pred_request = PredictionRequest(
                reaction=reaction_input,
                model_type=request.model_type,
                uncertainty_method=request.uncertainty_method,
                n_samples=50  # Reduce for batch
            )
            
            # Predict
            result = await predict_reaction_rate(pred_request)
            predictions.append(result)
        
        except Exception as e:
            errors += 1
            # Add error placeholder
            predictions.append(
                PredictionResponse(
                    prediction=0.0,
                    model_used=request.model_type.value,
                    metadata={
                        "error": str(e),
                        "index": i
                    }
                )
            )
    
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(request.reactions),
        errors=errors
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
