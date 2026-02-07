import { TrendingUp, AlertTriangle, CheckCircle, Info } from 'lucide-react'
import { PredictionResponse } from '@/lib/api'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from 'recharts'

interface PredictionResultsProps {
  result: PredictionResponse
}

const PredictionResults = ({ result }: PredictionResultsProps) => {
  const hasUncertainty = result.uncertainty !== undefined && result.uncertainty !== null

  // Generate data for uncertainty visualization
  const uncertaintyData = hasUncertainty
    ? Array.from({ length: 50 }, (_, i) => {
        const x = i / 50
        const mean = result.prediction
        const std = Math.sqrt(result.uncertainty!.total)
        return {
          x,
          mean,
          lower: mean - 1.96 * std,
          upper: mean + 1.96 * std,
        }
      })
    : []

  return (
    <div className="space-y-4">
      {/* Main Prediction Card */}
      <div className="bg-gradient-to-r from-primary-50 to-purple-50 rounded-lg shadow-lg border border-primary-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-primary-600" />
            Prediction Result
          </h3>
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
            <CheckCircle className="w-3 h-3 mr-1" />
            {result.model_used.toUpperCase()}
          </span>
        </div>

        <div className="text-center py-6">
          <p className="text-sm text-gray-600 mb-2">Predicted Reaction Rate</p>
          <p className="text-5xl font-bold text-primary-600">
            {result.prediction.toFixed(4)}
          </p>
          <p className="text-sm text-gray-500 mt-2">mol/L·s</p>
        </div>

        {hasUncertainty && result.uncertainty && (
          <div className="mt-6 pt-6 border-t border-primary-200">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <p className="text-xs text-gray-600 mb-1">95% Confidence Interval</p>
                <p className="text-sm font-medium text-gray-900">
                  [{result.uncertainty.confidence_interval_95[0].toFixed(4)},{' '}
                  {result.uncertainty.confidence_interval_95[1].toFixed(4)}]
                </p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-600 mb-1">Total Uncertainty</p>
                <p className="text-sm font-medium text-gray-900">
                  ±{result.uncertainty.total.toFixed(4)}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Uncertainty Breakdown */}
      {hasUncertainty && result.uncertainty && (
        <div className="bg-white rounded-lg shadow p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4 flex items-center">
            <AlertTriangle className="w-4 h-4 mr-2 text-yellow-600" />
            Uncertainty Analysis
          </h4>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Epistemic (Model)</span>
                <span className="font-medium text-gray-900">
                  {result.uncertainty.epistemic.toFixed(4)}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full"
                  style={{
                    width: `${(result.uncertainty.epistemic / result.uncertainty.total) * 100}%`,
                  }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Reducible with more training data
              </p>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Aleatoric (Data)</span>
                <span className="font-medium text-gray-900">
                  {result.uncertainty.aleatoric.toFixed(4)}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-yellow-600 h-2 rounded-full"
                  style={{
                    width: `${(result.uncertainty.aleatoric / result.uncertainty.total) * 100}%`,
                  }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Irreducible (inherent noise)
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Confidence Visualization */}
      {hasUncertainty && uncertaintyData.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4 flex items-center">
            <Info className="w-4 h-4 mr-2 text-blue-600" />
            Confidence Distribution
          </h4>

          <ResponsiveContainer width="100%" height={200}>
            <ComposedChart data={uncertaintyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis hide />
              <YAxis
                domain={['dataMin - 0.1', 'dataMax + 0.1']}
                tickFormatter={(value) => value.toFixed(3)}
              />
              <Tooltip
                formatter={(value: any) => value.toFixed(4)}
                labelFormatter={() => 'Prediction Range'}
              />
              <Area
                type="monotone"
                dataKey="upper"
                stroke="none"
                fill="#bae6fd"
                fillOpacity={0.5}
              />
              <Area
                type="monotone"
                dataKey="lower"
                stroke="none"
                fill="#ffffff"
              />
              <Line
                type="monotone"
                dataKey="mean"
                stroke="#0284c7"
                strokeWidth={2}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>

          <p className="text-xs text-gray-500 text-center mt-2">
            Shaded area represents 95% confidence interval
          </p>
        </div>
      )}

      {/* Metadata */}
      {result.metadata && Object.keys(result.metadata).length > 0 && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Additional Information</h4>
          <div className="text-xs text-gray-600 space-y-1">
            {result.metadata.note && (
              <p className="text-yellow-700 bg-yellow-50 p-2 rounded">
                ⚠️ {result.metadata.note}
              </p>
            )}
            {result.metadata.temperature && (
              <p>Temperature: {result.metadata.temperature}°C</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default PredictionResults
