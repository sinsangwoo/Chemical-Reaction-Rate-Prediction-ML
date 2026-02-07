"""Pydantic models for API request/response validation.

Defines schemas for:
- Input validation (SMILES, reaction conditions)
- Output formatting (predictions, uncertainty estimates)
- Error responses
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Literal
from enum import Enum


class ModelType(str, Enum):
    """Available model types."""
    RANDOM_FOREST = "random_forest"
    GCN = "gcn"
    GAT = "gat"
    GIN = "gin"
    MPNN = "mpnn"
    MC_DROPOUT = "mc_dropout"
    BAYESIAN_GNN = "bayesian_gnn"
    ENSEMBLE = "ensemble"


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification methods."""
    NONE = "none"
    MC_DROPOUT = "mc_dropout"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"
    CONFORMAL = "conformal"


class ReactionConditions(BaseModel):
    """Reaction conditions for prediction.
    
    Attributes:
        temperature: Temperature in Celsius
        pressure: Pressure in atm (optional)
        catalyst: Catalyst SMILES or name (optional)
        solvent: Solvent name or SMILES (optional)
        time: Reaction time in hours (optional)
    """
    temperature: float = Field(
        ..., 
        ge=-273.15, 
        le=500.0,
        description="Temperature in Celsius (-273.15 to 500)"
    )
    pressure: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=1000.0,
        description="Pressure in atm (0 to 1000)"
    )
    catalyst: Optional[str] = Field(
        None,
        description="Catalyst SMILES or name"
    )
    solvent: Optional[str] = Field(
        None,
        description="Solvent name or SMILES"
    )
    time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Reaction time in hours"
    )


class MoleculeInput(BaseModel):
    """Single molecule input.
    
    Attributes:
        smiles: SMILES string
        name: Optional molecule name
    """
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="SMILES notation of molecule"
    )
    name: Optional[str] = Field(
        None,
        description="Optional molecule name"
    )
    
    @validator('smiles')
    def validate_smiles_format(cls, v):
        """Basic SMILES validation."""
        if not v or not v.strip():
            raise ValueError("SMILES cannot be empty")
        # Add more validation as needed
        return v.strip()


class ReactionInput(BaseModel):
    """Reaction input for rate prediction.
    
    Attributes:
        reactants: List of reactant SMILES
        products: List of product SMILES
        conditions: Reaction conditions
        agents: Optional catalysts/reagents (not consumed)
    """
    reactants: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Reactant SMILES strings"
    )
    products: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Product SMILES strings"
    )
    conditions: ReactionConditions = Field(
        ...,
        description="Reaction conditions"
    )
    agents: Optional[List[str]] = Field(
        None,
        max_items=5,
        description="Optional agents (catalysts, etc.)"
    )


class PredictionRequest(BaseModel):
    """Prediction request with model selection.
    
    Attributes:
        reaction: Reaction specification
        model_type: Which model to use
        uncertainty_method: How to estimate uncertainty
        n_samples: Number of samples for uncertainty (MC methods)
    """
    reaction: ReactionInput = Field(
        ...,
        description="Reaction to predict"
    )
    model_type: ModelType = Field(
        ModelType.GIN,
        description="Model type to use"
    )
    uncertainty_method: UncertaintyMethod = Field(
        UncertaintyMethod.MC_DROPOUT,
        description="Uncertainty estimation method"
    )
    n_samples: int = Field(
        100,
        ge=10,
        le=1000,
        description="Number of MC samples for uncertainty"
    )


class UncertaintyEstimate(BaseModel):
    """Uncertainty estimate.
    
    Attributes:
        epistemic: Model uncertainty (reducible)
        aleatoric: Data uncertainty (irreducible)
        total: Total uncertainty
        confidence_interval_95: 95% confidence interval [lower, upper]
    """
    epistemic: float = Field(
        ...,
        description="Model/epistemic uncertainty"
    )
    aleatoric: float = Field(
        ...,
        description="Data/aleatoric uncertainty"
    )
    total: float = Field(
        ...,
        description="Total uncertainty"
    )
    confidence_interval_95: List[float] = Field(
        ...,
        description="95% confidence interval [lower, upper]"
    )


class PredictionResponse(BaseModel):
    """Prediction result with uncertainty.
    
    Attributes:
        prediction: Predicted reaction rate (mol/L·s)
        uncertainty: Uncertainty estimate (if requested)
        model_used: Which model was used
        metadata: Additional information
    """
    prediction: float = Field(
        ...,
        description="Predicted reaction rate (mol/L·s)"
    )
    uncertainty: Optional[UncertaintyEstimate] = Field(
        None,
        description="Uncertainty estimate"
    )
    model_used: str = Field(
        ...,
        description="Model type used for prediction"
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request.
    
    Attributes:
        reactions: List of reactions to predict
        model_type: Model to use for all predictions
        uncertainty_method: Uncertainty method
    """
    reactions: List[ReactionInput] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Reactions to predict (max 100)"
    )
    model_type: ModelType = Field(
        ModelType.GIN,
        description="Model type"
    )
    uncertainty_method: UncertaintyMethod = Field(
        UncertaintyMethod.NONE,
        description="Uncertainty method (None for batch speed)"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction results.
    
    Attributes:
        predictions: List of predictions
        total: Total number of predictions
        errors: Number of errors
    """
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions"
    )
    total: int = Field(
        ...,
        description="Total predictions requested"
    )
    errors: int = Field(
        0,
        description="Number of failed predictions"
    )


class MoleculeValidationResponse(BaseModel):
    """SMILES validation result.
    
    Attributes:
        is_valid: Whether SMILES is valid
        smiles: Canonicalized SMILES (if valid)
        errors: Validation errors (if any)
        properties: Basic molecular properties
    """
    is_valid: bool = Field(
        ...,
        description="Whether SMILES is valid"
    )
    smiles: Optional[str] = Field(
        None,
        description="Canonicalized SMILES"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Validation errors"
    )
    properties: Optional[Dict] = Field(
        None,
        description="Molecular properties"
    )


class HealthResponse(BaseModel):
    """API health check response.
    
    Attributes:
        status: Service status
        version: API version
        models_loaded: Available models
        uptime_seconds: Server uptime
    """
    status: str = Field(
        ...,
        description="Service status"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    models_loaded: List[str] = Field(
        ...,
        description="Available model types"
    )
    uptime_seconds: float = Field(
        ...,
        description="Server uptime in seconds"
    )


class ErrorResponse(BaseModel):
    """Error response.
    
    Attributes:
        error: Error type
        message: Error message
        details: Additional error details
    """
    error: str = Field(
        ...,
        description="Error type"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[Dict] = Field(
        None,
        description="Additional error details"
    )
