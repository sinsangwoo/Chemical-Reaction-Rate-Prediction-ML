# Chemical Reaction ML API

REST API for molecular property and reaction rate prediction with uncertainty quantification.

## Quick Start

### Installation

```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# For full functionality (GNN models)
pip install torch torch-geometric
```

### Run Server

```bash
# Development mode (with auto-reload)
uvicorn api.main:app --reload --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

API will be available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": ["random_forest", "gcn", "gat", "gin", "mpnn"],
  "uptime_seconds": 3600.5
}
```

### List Models

```bash
GET /models
```

Response:
```json
{
  "traditional": {
    "random_forest": {
      "description": "RandomForest regressor",
      "uncertainty": false,
      "speed": "fast"
    }
  },
  "gnn": {
    "gin": {
      "description": "Graph Isomorphism Network",
      "uncertainty": false,
      "speed": "medium"
    }
  },
  "bayesian": {
    "mc_dropout": {
      "description": "Monte Carlo Dropout",
      "uncertainty": true,
      "speed": "medium"
    }
  }
}
```

### Validate SMILES

```bash
POST /validate/smiles
```

Request:
```json
{
  "smiles": "c1ccccc1",
  "name": "benzene"
}
```

Response:
```json
{
  "is_valid": true,
  "smiles": "c1ccccc1",
  "errors": [],
  "properties": {
    "length": 8,
    "num_atoms": 6,
    "num_rings": 1,
    "has_aromatic": true,
    "num_branches": 0,
    "estimated_mw": 78.0,
    "atom_counts": {"C": 6}
  }
}
```

### Predict Reaction Rate

```bash
POST /predict
```

Request:
```json
{
  "reaction": {
    "reactants": ["CCO", "CC(=O)O"],
    "products": ["CCOC(=O)C"],
    "conditions": {
      "temperature": 80.0,
      "pressure": 1.0,
      "catalyst": "H2SO4",
      "solvent": "water"
    },
    "agents": ["[H+]"]
  },
  "model_type": "gin",
  "uncertainty_method": "mc_dropout",
  "n_samples": 100
}
```

Response:
```json
{
  "prediction": 0.4523,
  "uncertainty": {
    "epistemic": 0.0234,
    "aleatoric": 0.0100,
    "total": 0.0334,
    "confidence_interval_95": [0.3964, 0.5082]
  },
  "model_used": "gin",
  "metadata": {
    "reactants": ["CCO", "CC(=O)O"],
    "products": ["CCOC(=O)C"],
    "temperature": 80.0
  }
}
```

### Batch Prediction

```bash
POST /predict/batch
```

Request:
```json
{
  "reactions": [
    {
      "reactants": ["CCO"],
      "products": ["CC=O"],
      "conditions": {"temperature": 100.0}
    },
    {
      "reactants": ["c1ccccc1"],
      "products": ["c1ccc(Br)cc1"],
      "conditions": {"temperature": 50.0}
    }
  ],
  "model_type": "gin",
  "uncertainty_method": "none"
}
```

Response:
```json
{
  "predictions": [
    {
      "prediction": 0.4523,
      "uncertainty": null,
      "model_used": "gin",
      "metadata": {}
    },
    {
      "prediction": 0.3214,
      "uncertainty": null,
      "model_used": "gin",
      "metadata": {}
    }
  ],
  "total": 2,
  "errors": 0
}
```

---

## Python Client Examples

### Basic Usage

```python
import requests

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Validate SMILES
response = requests.post(
    f"{BASE_URL}/validate/smiles",
    json={"smiles": "c1ccccc1", "name": "benzene"}
)
print(response.json())
```

### Predict Reaction Rate

```python
import requests

BASE_URL = "http://localhost:8000"

request = {
    "reaction": {
        "reactants": ["CCO", "CC(=O)O"],
        "products": ["CCOC(=O)C"],
        "conditions": {
            "temperature": 80.0,
            "catalyst": "H2SO4"
        }
    },
    "model_type": "gin",
    "uncertainty_method": "mc_dropout",
    "n_samples": 100
}

response = requests.post(f"{BASE_URL}/predict", json=request)
result = response.json()

print(f"Prediction: {result['prediction']:.4f}")
print(f"95% CI: {result['uncertainty']['confidence_interval_95']}")
```

### Batch Prediction

```python
import requests
import pandas as pd

BASE_URL = "http://localhost:8000"

# Load reactions from CSV
df = pd.read_csv("reactions.csv")

reactions = [
    {
        "reactants": [row["reactant_smiles"]],
        "products": [row["product_smiles"]],
        "conditions": {"temperature": row["temperature"]}
    }
    for _, row in df.iterrows()
]

request = {
    "reactions": reactions[:100],  # Max 100
    "model_type": "gin",
    "uncertainty_method": "none"
}

response = requests.post(f"{BASE_URL}/predict/batch", json=request)
results = response.json()

# Add predictions to DataFrame
df["predicted_rate"] = [p["prediction"] for p in results["predictions"]]
```

---

## JavaScript/TypeScript Client

### Fetch API

```javascript
const BASE_URL = "http://localhost:8000";

// Predict reaction rate
async function predictReactionRate(reaction) {
  const response = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      reaction: {
        reactants: ["CCO", "CC(=O)O"],
        products: ["CCOC(=O)C"],
        conditions: { temperature: 80.0 }
      },
      model_type: "gin",
      uncertainty_method: "mc_dropout"
    })
  });
  
  return await response.json();
}

// Usage
predictReactionRate().then(result => {
  console.log(`Prediction: ${result.prediction}`);
  console.log(`Uncertainty: ${result.uncertainty.total}`);
});
```

### Axios

```typescript
import axios from 'axios';

const BASE_URL = "http://localhost:8000";

interface PredictionResponse {
  prediction: number;
  uncertainty?: {
    epistemic: number;
    aleatoric: number;
    total: number;
    confidence_interval_95: [number, number];
  };
  model_used: string;
  metadata: Record<string, any>;
}

async function predictRate(): Promise<PredictionResponse> {
  const response = await axios.post(`${BASE_URL}/predict`, {
    reaction: {
      reactants: ["CCO", "CC(=O)O"],
      products: ["CCOC(=O)C"],
      conditions: { temperature: 80.0 }
    },
    model_type: "gin",
    uncertainty_method: "mc_dropout"
  });
  
  return response.data;
}
```

---

## cURL Examples

### Health Check

```bash
curl http://localhost:8000/health
```

### Validate SMILES

```bash
curl -X POST http://localhost:8000/validate/smiles \
  -H "Content-Type: application/json" \
  -d '{"smiles": "c1ccccc1", "name": "benzene"}'
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "reaction": {
      "reactants": ["CCO", "CC(=O)O"],
      "products": ["CCOC(=O)C"],
      "conditions": {"temperature": 80.0}
    },
    "model_type": "gin",
    "uncertainty_method": "mc_dropout"
  }'
```

---

## Docker Deployment

### Build Image

```bash
docker build -t chem-ml-api .
```

### Run Container

```bash
docker run -p 8000:8000 chem-ml-api
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models
    volumes:
      - ./models:/models
    restart: unless-stopped
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "ValidationError",
  "message": "Invalid SMILES syntax",
  "details": {
    "field": "reactants[0]",
    "value": "invalid_smiles"
  }
}
```

### Common Errors

| Status Code | Error | Description |
|-------------|-------|-------------|
| 400 | Bad Request | Invalid input (SMILES, parameters) |
| 422 | Validation Error | Pydantic validation failed |
| 500 | Internal Server Error | Server error |
| 501 | Not Implemented | Model not available (missing PyTorch) |

---

## Performance

### Benchmarks

| Endpoint | Latency (p50) | Throughput |
|----------|---------------|------------|
| /validate/smiles | 5 ms | 2000 req/s |
| /predict (RF) | 20 ms | 500 req/s |
| /predict (GNN) | 50 ms | 200 req/s |
| /predict (Bayesian) | 500 ms | 20 req/s |
| /predict/batch (100) | 2 s | - |

### Optimization Tips

1. **Model Caching**: Models loaded once and cached
2. **Batch Requests**: Use `/predict/batch` for multiple predictions
3. **Reduce Uncertainty Samples**: Lower `n_samples` for faster inference
4. **Production Server**: Use `gunicorn` with multiple workers
5. **GPU Acceleration**: CUDA support for GNN models

---

## Security

### Production Checklist

- [ ] Set CORS allowed origins (not `*`)
- [ ] Add API key authentication
- [ ] Rate limiting
- [ ] HTTPS/TLS
- [ ] Input sanitization
- [ ] Request size limits
- [ ] Monitoring and logging

### Rate Limiting Example

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_reaction_rate(request: Request, ...):
    ...
```

---

## Monitoring

### Prometheus Metrics

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Metrics available at `/metrics`

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

---

## Troubleshooting

### Issue: Models not loading

**Solution**: Check PyTorch installation
```bash
pip install torch torch-geometric
```

### Issue: CORS errors

**Solution**: Add frontend origin to allowed origins
```python
allow_origins=["http://localhost:3000"]  # React app
```

### Issue: Slow predictions

**Solution**: 
1. Use simpler model (RF instead of GNN)
2. Reduce `n_samples` for uncertainty
3. Enable GPU acceleration
4. Use batch endpoint

---

## Development

### Run Tests

```bash
pytest tests/test_api.py -v
```

### Hot Reload

```bash
uvicorn api.main:app --reload
```

### Generate OpenAPI Spec

```bash
curl http://localhost:8000/openapi.json > openapi.json
```

---

## License

MIT License - see LICENSE file
