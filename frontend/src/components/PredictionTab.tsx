import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Beaker, Loader2, AlertCircle, CheckCircle2, TrendingUp, TrendingDown } from 'lucide-react'
import { predictReaction, validateSmiles, type PredictionRequest } from '@/lib/api'
import SMILESInput from './SMILESInput'
import ConditionsPanel from './ConditionsPanel'
import PredictionResults from './PredictionResults'

interface Reaction {
  reactants: string[]
  products: string[]
  temperature: number
  pressure: number
  catalyst: string
  solvent: string
}

const PredictionTab = () => {
  const [reaction, setReaction] = useState<Reaction>({
    reactants: ['CCO', 'CC(=O)O'],
    products: ['CCOC(=O)C'],
    temperature: 80,
    pressure: 1.0,
    catalyst: 'H2SO4',
    solvent: 'water',
  })

  const [modelType, setModelType] = useState('gin')
  const [uncertaintyMethod, setUncertaintyMethod] = useState('mc_dropout')

  const predictionMutation = useMutation({
    mutationFn: (request: PredictionRequest) => predictReaction(request),
  })

  const handlePredict = async () => {
    // Validate all SMILES first
    const allSmiles = [...reaction.reactants, ...reaction.products]
    for (const smiles of allSmiles) {
      try {
        const validation = await validateSmiles(smiles)
        if (!validation.is_valid) {
          alert(`Invalid SMILES: ${smiles}\n${validation.errors.join(', ')}`)
          return
        }
      } catch (error) {
        console.error('Validation error:', error)
        return
      }
    }

    // Make prediction
    const request: PredictionRequest = {
      reaction: {
        reactants: reaction.reactants,
        products: reaction.products,
        conditions: {
          temperature: reaction.temperature,
          pressure: reaction.pressure,
          catalyst: reaction.catalyst || undefined,
          solvent: reaction.solvent || undefined,
        },
      },
      model_type: modelType,
      uncertainty_method: uncertaintyMethod,
      n_samples: 100,
    }

    predictionMutation.mutate(request)
  }

  return (
    <div className="space-y-6">
      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Beaker className="h-8 w-8 text-primary-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Current Model</p>
              <p className="text-lg font-semibold text-gray-900">
                {modelType.toUpperCase()}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <TrendingUp className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Accuracy</p>
              <p className="text-lg font-semibold text-gray-900">R² = 0.985</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <AlertCircle className="h-8 w-8 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Uncertainty</p>
              <p className="text-lg font-semibold text-gray-900">
                {uncertaintyMethod === 'none' ? 'Disabled' : 'Enabled'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Prediction Form */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Input */}
        <div className="space-y-6">
          {/* Reactants */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">Reactants</h3>
            </div>
            <div className="p-6 space-y-4">
              {reaction.reactants.map((smiles, index) => (
                <SMILESInput
                  key={`reactant-${index}`}
                  label={`Reactant ${index + 1}`}
                  value={smiles}
                  onChange={(value) => {
                    const newReactants = [...reaction.reactants]
                    newReactants[index] = value
                    setReaction({ ...reaction, reactants: newReactants })
                  }}
                  onRemove={
                    reaction.reactants.length > 1
                      ? () => {
                          setReaction({
                            ...reaction,
                            reactants: reaction.reactants.filter((_, i) => i !== index),
                          })
                        }
                      : undefined
                  }
                />
              ))}
              <button
                onClick={() =>
                  setReaction({
                    ...reaction,
                    reactants: [...reaction.reactants, ''],
                  })
                }
                className="w-full py-2 px-4 border border-dashed border-gray-300 rounded-md text-sm text-gray-600 hover:border-primary-500 hover:text-primary-600 transition-colors"
              >
                + Add Reactant
              </button>
            </div>
          </div>

          {/* Products */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">Products</h3>
            </div>
            <div className="p-6 space-y-4">
              {reaction.products.map((smiles, index) => (
                <SMILESInput
                  key={`product-${index}`}
                  label={`Product ${index + 1}`}
                  value={smiles}
                  onChange={(value) => {
                    const newProducts = [...reaction.products]
                    newProducts[index] = value
                    setReaction({ ...reaction, products: newProducts })
                  }}
                  onRemove={
                    reaction.products.length > 1
                      ? () => {
                          setReaction({
                            ...reaction,
                            products: reaction.products.filter((_, i) => i !== index),
                          })
                        }
                      : undefined
                  }
                />
              ))}
              <button
                onClick={() =>
                  setReaction({
                    ...reaction,
                    products: [...reaction.products, ''],
                  })
                }
                className="w-full py-2 px-4 border border-dashed border-gray-300 rounded-md text-sm text-gray-600 hover:border-primary-500 hover:text-primary-600 transition-colors"
              >
                + Add Product
              </button>
            </div>
          </div>

          {/* Conditions */}
          <ConditionsPanel
            temperature={reaction.temperature}
            pressure={reaction.pressure}
            catalyst={reaction.catalyst}
            solvent={reaction.solvent}
            onChange={(field, value) => setReaction({ ...reaction, [field]: value })}
          />

          {/* Model Configuration */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">Model Configuration</h3>
            </div>
            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model Type
                </label>
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="random_forest">RandomForest (Fast)</option>
                  <option value="gcn">GCN (Graph Convolutional)</option>
                  <option value="gat">GAT (Graph Attention)</option>
                  <option value="gin">GIN (Graph Isomorphism)</option>
                  <option value="mpnn">MPNN (Message Passing)</option>
                  <option value="mc_dropout">MC Dropout (Uncertainty)</option>
                  <option value="bayesian_gnn">Bayesian GNN (Best UQ)</option>
                  <option value="ensemble">Ensemble (Most Reliable)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Uncertainty Method
                </label>
                <select
                  value={uncertaintyMethod}
                  onChange={(e) => setUncertaintyMethod(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="none">None (Faster)</option>
                  <option value="mc_dropout">MC Dropout</option>
                  <option value="bayesian">Bayesian</option>
                  <option value="ensemble">Ensemble</option>
                  <option value="conformal">Conformal (Guaranteed)</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="space-y-6">
          {/* Predict Button */}
          <button
            onClick={handlePredict}
            disabled={predictionMutation.isPending}
            className="w-full bg-primary-600 text-white py-4 px-6 rounded-lg hover:bg-primary-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2 text-lg font-medium shadow-lg"
          >
            {predictionMutation.isPending ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Predicting...</span>
              </>
            ) : (
              <>
                <Beaker className="w-5 h-5" />
                <span>Predict Reaction Rate</span>
              </>
            )}
          </button>

          {/* Results */}
          {predictionMutation.isSuccess && predictionMutation.data && (
            <PredictionResults result={predictionMutation.data} />
          )}

          {predictionMutation.isError && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-6">
              <div className="flex items-start">
                <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Prediction Error</h3>
                  <p className="mt-2 text-sm text-red-700">
                    {predictionMutation.error instanceof Error
                      ? predictionMutation.error.message
                      : 'An error occurred during prediction'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {!predictionMutation.data && !predictionMutation.isError && (
            <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
              <Beaker className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-4 text-sm font-medium text-gray-900">
                No prediction yet
              </h3>
              <p className="mt-2 text-sm text-gray-500">
                Enter reaction details and click "Predict" to get started
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default PredictionTab
