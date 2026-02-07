import axios from 'axios'

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth tokens here if needed
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error
      console.error('API Error:', error.response.data)
    } else if (error.request) {
      // No response received
      console.error('Network Error:', error.message)
    }
    return Promise.reject(error)
  }
)

// API Types
export interface ReactionConditions {
  temperature: number
  pressure?: number
  catalyst?: string
  solvent?: string
  time?: number
}

export interface ReactionInput {
  reactants: string[]
  products: string[]
  conditions: ReactionConditions
  agents?: string[]
}

export interface UncertaintyEstimate {
  epistemic: number
  aleatoric: number
  total: number
  confidence_interval_95: [number, number]
}

export interface PredictionResponse {
  prediction: number
  uncertainty?: UncertaintyEstimate
  model_used: string
  metadata?: Record<string, any>
}

export interface PredictionRequest {
  reaction: ReactionInput
  model_type: string
  uncertainty_method: string
  n_samples?: number
}

export interface MoleculeValidationResponse {
  is_valid: boolean
  smiles?: string
  errors: string[]
  properties?: {
    length: number
    num_atoms: number
    num_rings: number
    has_aromatic: boolean
    num_branches: number
    estimated_mw: number
    atom_counts: Record<string, number>
  }
}

export interface HealthResponse {
  status: string
  version: string
  models_loaded: string[]
  uptime_seconds: number
}

// API Functions
export const healthCheck = async (): Promise<HealthResponse> => {
  const { data } = await api.get('/health')
  return data
}

export const validateSmiles = async (smiles: string): Promise<MoleculeValidationResponse> => {
  const { data } = await api.post('/validate/smiles', { smiles })
  return data
}

export const predictReaction = async (
  request: PredictionRequest
): Promise<PredictionResponse> => {
  const { data } = await api.post('/predict', request)
  return data
}

export const listModels = async () => {
  const { data } = await api.get('/models')
  return data
}
