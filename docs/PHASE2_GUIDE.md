# Phase 2: Real Chemistry & SMILES Integration

## Overview

Phase 2 transforms the project from synthetic toy data to **real chemical reaction representations** using industry-standard SMILES notation and real datasets like USPTO.

---

## What's New in Phase 2?

### 1. SMILES Support

**SMILES** (Simplified Molecular Input Line Entry System) is the standard way to represent molecules as text.

```python
from src.data.smiles_parser import SMILESParser

parser = SMILESParser()

# Parse ethanol
features = parser.extract_features("CCO")
print(features)
# {
#   'atom_counts': {'C': 2, 'O': 1},
#   'estimated_mw': 46.07,
#   'has_aromatic': False,
#   ...
# }
```

**Reaction SMILES** format:
```
reactants>agents>products
CCO.CC(=O)O>H2SO4>CCOC(=O)C
# Ethanol + Acetic Acid -> Ethyl Acetate (with H2SO4 catalyst)
```

### 2. Real Chemical Reactions

```python
from src.data.reaction_dataset import ChemicalReaction, ReactionConditions

conditions = ReactionConditions(
    temperature=80.0,  # °C
    catalyst="H2SO4",
    solvent="toluene",
    time=3600  # seconds
)

reaction = ChemicalReaction(
    reaction_id="rxn_001",
    reactants=["CCO", "CC(=O)O"],  # SMILES
    products=["CCOC(=O)C"],
    conditions=conditions,
    yield_percentage=85.0
)
```

### 3. USPTO Dataset Integration

```python
from src.data.uspto_loader import USPTOLoader

loader = USPTOLoader()

# Generate synthetic USPTO-style data
dataset = loader.create_synthetic_dataset(num_reactions=1000)

# Or download real USPTO data (when available)
# path = loader.download_dataset('uspto_50k_sample')
# dataset = loader.load_from_csv(path)

dataset.save_dataset("data/processed/uspto_1k.json")
```

### 4. Molecular Feature Extraction

```python
from src.features.molecular_features import ReactionFeatureBuilder

builder = ReactionFeatureBuilder()
features_df, targets = builder.build_features(dataset.reactions)

# features_df: ML-ready features
# - Reactant molecular properties
# - Product molecular properties
# - Delta features (change)
# - Reaction conditions
```

---

## Key Components

### SMILESParser

**Purpose**: Parse and validate SMILES strings

**Capabilities**:
- Validate SMILES syntax
- Extract atom counts
- Estimate molecular weight
- Count rings and branches
- Detect aromatic compounds

### ReactionSMILES

**Purpose**: Handle reaction SMILES notation

**Capabilities**:
- Parse reaction SMILES into components
- Validate reaction syntax
- Extract reaction features

### ReactionDataset

**Purpose**: Manage collections of chemical reactions

**Capabilities**:
- Load/save JSON and CSV formats
- Filter by conditions (temperature, catalyst, etc.)
- Calculate statistics
- Convert to/from DataFrames

### USPTOLoader

**Purpose**: Integrate with USPTO chemical reaction database

**Capabilities**:
- Download USPTO datasets
- Parse USPTO CSV format
- Generate synthetic USPTO-style data
- Extract reaction conditions

### MolecularFeatureExtractor

**Purpose**: Convert molecules to ML features

**Features Extracted**:
- Atom composition (C, N, O, etc.)
- Structural properties (rings, branches)
- Molecular weight
- Complexity metrics (atom diversity)
- Aromatic character

---

## Usage Examples

### Example 1: Parse and Analyze Molecules

```python
from src.data.smiles_parser import SMILESParser

parser = SMILESParser()

molecules = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O"
}

for name, smiles in molecules.items():
    features = parser.extract_features(smiles)
    print(f"{name}: {features['atom_counts']}, MW ~{features['estimated_mw']:.1f}")
```

### Example 2: Build a Reaction Dataset

```python
from src.data.reaction_dataset import ReactionDataset, ChemicalReaction, ReactionConditions
from src.data.smiles_parser import ReactionSMILES

dataset = ReactionDataset()
parser = ReactionSMILES()

# Add a reaction
rxn_smiles = "CCO.CC(=O)O>>CCOC(=O)C"
parsed = parser.parse_reaction(rxn_smiles)

conditions = ReactionConditions(temperature=80.0, catalyst="H2SO4")
reaction = ChemicalReaction(
    reaction_id="rxn_001",
    reactants=parsed["reactants"],
    products=parsed["products"],
    conditions=conditions,
    yield_percentage=85.0
)

dataset.add_reaction(reaction)
dataset.save_dataset("my_reactions.json")
```

### Example 3: Train ML Model on Real Chemistry

```python
from src.data.uspto_loader import USPTOLoader
from src.features.molecular_features import ReactionFeatureBuilder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Generate synthetic USPTO data
loader = USPTOLoader()
dataset = loader.create_synthetic_dataset(num_reactions=500)

# Extract features
builder = ReactionFeatureBuilder()
X, y = builder.build_features(dataset.reactions)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"R² Score: {score:.3f}")
```

---

## Data Formats

### JSON Format

```json
{
  "reactions": [
    {
      "reaction_id": "rxn_001",
      "reactants": ["CCO", "CC(=O)O"],
      "products": ["CCOC(=O)C"],
      "agents": ["H2SO4"],
      "conditions": {
        "temperature": 80.0,
        "catalyst": "H2SO4",
        "solvent": "toluene"
      },
      "yield_percentage": 85.0,
      "source": "USPTO"
    }
  ]
}
```

### CSV Format

```csv
reaction_id,reaction_smiles,temperature,catalyst,yield,source
rxn_001,"CCO.CC(=O)O>>CCOC(=O)C",80.0,H2SO4,85.0,USPTO
```

---

## Running the Demo

```bash
python examples/phase2_demo.py
```

This demonstrates:
1. SMILES parsing for molecules
2. Reaction SMILES parsing
3. Dataset creation and management
4. USPTO-style data generation
5. Feature extraction for ML

---

## Testing

```bash
# Run Phase 2 tests
pytest tests/test_smiles_parser.py -v
pytest tests/test_reaction_dataset.py -v

# Run all tests
pytest tests/ -v
```

---

## Comparison: Phase 1 vs Phase 2

| Aspect | Phase 1 | Phase 2 |
|--------|---------|----------|
| **Data** | Synthetic (temp, conc, catalyst flag) | Real chemistry (SMILES, reactions) |
| **Molecules** | Not represented | SMILES notation |
| **Reactions** | Implicit | Explicit (reactants→products) |
| **Conditions** | 3 features | Rich conditions (temp, catalyst, solvent, time) |
| **Features** | 3 numerical | 30+ molecular descriptors |
| **Datasets** | Generated only | USPTO integration |
| **Format** | CSV | JSON + CSV |
| **Realism** | Low (toy problem) | High (real chemistry) |

---

## Next Steps (Phase 3)

Phase 2 provides the foundation for:

1. **Graph Neural Networks**: Represent molecules as graphs
2. **RDKit Integration**: Advanced molecular descriptors
3. **Transformer Models**: Sequence-based reaction prediction
4. **Transfer Learning**: Pre-trained chemistry models

---

## References

- **SMILES**: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
- **USPTO Dataset**: Lowe, D. (2017). Chemical reactions from US patents (1976-Sep2016)
- **ORDerly**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11094788/
- **USPTO-LLM**: https://zenodo.org/records/14396156

---

## Troubleshooting

### Issue: SMILES validation fails

**Solution**: Ensure SMILES syntax is correct. Common issues:
- Unbalanced parentheses: `(C(C)` → `C(C)C`
- Invalid atoms: Check atom symbols
- Too long: SMILES >500 chars may be rejected

### Issue: Feature extraction fails

**Solution**: Check that:
- All reactants/products are valid SMILES
- Reaction conditions are properly set
- Dataset isn't empty

---

**Phase 2 brings real chemistry to the project!** 🧪
