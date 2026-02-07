import { useQuery } from '@tanstack/react-query'
import { Brain, Zap, Target, Clock, CheckCircle2, XCircle, Loader2 } from 'lucide-react'
import { listModels, healthCheck } from '@/lib/api'

const ModelsTab = () => {
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: healthCheck,
    refetchInterval: 10000, // Refresh every 10s
  })

  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ['models'],
    queryFn: listModels,
  })

  const modelCategories = [
    {
      name: 'Traditional ML',
      key: 'traditional',
      icon: Target,
      color: 'blue',
    },
    {
      name: 'Graph Neural Networks',
      key: 'gnn',
      icon: Brain,
      color: 'purple',
    },
    {
      name: 'Bayesian Methods',
      key: 'bayesian',
      icon: Zap,
      color: 'yellow',
    },
  ]

  if (healthLoading || modelsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* System Status */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
          <CheckCircle2 className="w-5 h-5 mr-2 text-green-600" />
          System Status
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <span className="text-sm font-medium text-gray-600">Status</span>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              {health?.status || 'Unknown'}
            </span>
          </div>
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <span className="text-sm font-medium text-gray-600">Version</span>
            <span className="text-sm font-semibold text-gray-900">
              {health?.version || 'N/A'}
            </span>
          </div>
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <span className="text-sm font-medium text-gray-600">Uptime</span>
            <span className="text-sm font-semibold text-gray-900">
              {health?.uptime_seconds
                ? `${Math.floor(health.uptime_seconds / 3600)}h ${Math.floor(
                    (health.uptime_seconds % 3600) / 60
                  )}m`
                : 'N/A'}
            </span>
          </div>
        </div>
      </div>

      {/* Models by Category */}
      {modelCategories.map((category) => {
        const Icon = category.icon
        const categoryModels = models?.[category.key] || {}
        const isAvailable = typeof categoryModels === 'object' && !categoryModels.note

        return (
          <div key={category.key} className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900 flex items-center">
                <Icon className={`w-5 h-5 mr-2 text-${category.color}-600`} />
                {category.name}
              </h3>
            </div>
            <div className="p-6">
              {!isAvailable ? (
                <div className="text-center py-8 text-gray-500">
                  <XCircle className="w-12 h-12 mx-auto mb-2 text-gray-400" />
                  <p>{categoryModels.note || 'Not available'}</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(categoryModels).map(([name, info]: [string, any]) => (
                    <div
                      key={name}
                      className="border border-gray-200 rounded-lg p-4 hover:border-primary-500 transition-colors"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-medium text-gray-900">
                          {name.toUpperCase()}
                        </h4>
                        {info.uncertainty ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                            UQ
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
                            Standard
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-600 mb-3">
                        {info.description}
                      </p>
                      <div className="flex items-center justify-between text-xs">
                        <span className="flex items-center text-gray-500">
                          <Clock className="w-3 h-3 mr-1" />
                          {info.speed}
                        </span>
                        {health?.models_loaded?.includes(name) ? (
                          <CheckCircle2 className="w-4 h-4 text-green-600" />
                        ) : (
                          <XCircle className="w-4 h-4 text-gray-400" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )
      })}

      {/* Model Recommendations */}
      <div className="bg-gradient-to-r from-primary-50 to-purple-50 rounded-lg p-6 border border-primary-200">
        <h3 className="text-lg font-medium text-gray-900 mb-4">🎯 Model Selection Guide</h3>
        <div className="space-y-3">
          <div className="flex items-start">
            <Zap className="w-5 h-5 text-yellow-600 mr-3 mt-0.5" />
            <div>
              <h4 className="font-medium text-gray-900">For Speed</h4>
              <p className="text-sm text-gray-600">
                Use <strong>RandomForest</strong> - Fastest predictions (~20ms)
              </p>
            </div>
          </div>
          <div className="flex items-start">
            <Target className="w-5 h-5 text-blue-600 mr-3 mt-0.5" />
            <div>
              <h4 className="font-medium text-gray-900">For Accuracy</h4>
              <p className="text-sm text-gray-600">
                Use <strong>GIN</strong> - Best R² score (0.985)
              </p>
            </div>
          </div>
          <div className="flex items-start">
            <Brain className="w-5 h-5 text-purple-600 mr-3 mt-0.5" />
            <div>
              <h4 className="font-medium text-gray-900">For Reliability</h4>
              <p className="text-sm text-gray-600">
                Use <strong>Ensemble</strong> - Most robust with uncertainty
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ModelsTab
