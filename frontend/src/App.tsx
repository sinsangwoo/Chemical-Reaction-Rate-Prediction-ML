import { useState } from 'react'
import { Flask, Activity, BarChart3, Settings } from 'lucide-react'
import PredictionTab from './components/PredictionTab'
import AnalyticsTab from './components/AnalyticsTab'
import ModelsTab from './components/ModelsTab'

type Tab = 'prediction' | 'analytics' | 'models'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('prediction')

  const tabs = [
    { id: 'prediction' as Tab, name: 'Prediction', icon: Flask },
    { id: 'analytics' as Tab, name: 'Analytics', icon: BarChart3 },
    { id: 'models' as Tab, name: 'Models', icon: Settings },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <Activity className="w-8 h-8 text-primary-600" />
              <div>
                <h1 className="text-xl font-bold text-gray-900">
                  Chemical Reaction ML
                </h1>
                <p className="text-xs text-gray-500">AI-Powered Property Prediction</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                <span className="w-2 h-2 mr-1.5 bg-green-400 rounded-full animate-pulse"></span>
                Connected
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon
              const isActive = activeTab === tab.id
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    group inline-flex items-center py-4 px-1 border-b-2 font-medium text-sm
                    transition-colors duration-200
                    ${
                      isActive
                        ? 'border-primary-500 text-primary-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                  `}
                >
                  <Icon
                    className={`
                      -ml-0.5 mr-2 h-5 w-5
                      ${
                        isActive
                          ? 'text-primary-500'
                          : 'text-gray-400 group-hover:text-gray-500'
                      }
                    `}
                  />
                  {tab.name}
                </button>
              )
            })}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'prediction' && <PredictionTab />}
        {activeTab === 'analytics' && <AnalyticsTab />}
        {activeTab === 'models' && <ModelsTab />}
      </main>

      {/* Footer */}
      <footer className="mt-auto py-6 border-t border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            Built with React + FastAPI | Powered by GNN & Uncertainty Quantification
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
