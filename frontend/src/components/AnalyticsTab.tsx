import { BarChart3, TrendingUp, PieChart, Activity } from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
} from 'recharts'

const AnalyticsTab = () => {
  // Mock data for demonstration
  const modelPerformance = [
    { name: 'RandomForest', r2: 0.82, mae: 0.15, time: 0.02 },
    { name: 'GCN', r2: 0.91, mae: 0.11, time: 0.05 },
    { name: 'GAT', r2: 0.93, mae: 0.09, time: 0.06 },
    { name: 'GIN', r2: 0.985, mae: 0.05, time: 0.05 },
    { name: 'MPNN', r2: 0.94, mae: 0.08, time: 0.07 },
  ]

  const predictionHistory = [
    { date: '2026-02-01', predictions: 45 },
    { date: '2026-02-02', predictions: 67 },
    { date: '2026-02-03', predictions: 52 },
    { date: '2026-02-04', predictions: 89 },
    { date: '2026-02-05', predictions: 73 },
    { date: '2026-02-06', predictions: 94 },
    { date: '2026-02-07', predictions: 61 },
  ]

  const uncertaintyDistribution = [
    { name: 'Low (<0.05)', value: 65, color: '#10b981' },
    { name: 'Medium (0.05-0.15)', value: 25, color: '#f59e0b' },
    { name: 'High (>0.15)', value: 10, color: '#ef4444' },
  ]

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Total Predictions</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">481</p>
            </div>
            <Activity className="w-8 h-8 text-primary-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Avg Accuracy</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">R² 0.92</p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Avg Response</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">52ms</p>
            </div>
            <BarChart3 className="w-8 h-8 text-blue-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Models Active</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">8/8</p>
            </div>
            <PieChart className="w-8 h-8 text-purple-600" />
          </div>
        </div>
      </div>

      {/* Model Performance Comparison */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Model Performance Comparison</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={modelPerformance}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis yAxisId="left" orientation="left" domain={[0, 1]} />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Legend />
            <Bar yAxisId="left" dataKey="r2" fill="#0ea5e9" name="R² Score" />
            <Bar yAxisId="left" dataKey="mae" fill="#f59e0b" name="MAE" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Prediction History */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Prediction History (7 Days)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={predictionHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tickFormatter={(date) => date.slice(5)} />
              <YAxis />
              <Tooltip labelFormatter={(date) => `Date: ${date}`} />
              <Line
                type="monotone"
                dataKey="predictions"
                stroke="#0ea5e9"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Uncertainty Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Uncertainty Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <RechartsPieChart>
              <Pie
                data={uncertaintyDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(entry) => `${entry.name}: ${entry.value}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {uncertaintyDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </RechartsPieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Performance Insights */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 border border-blue-200">
        <h3 className="text-lg font-medium text-gray-900 mb-4">📊 Performance Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white rounded-lg p-4">
            <h4 className="font-medium text-sm text-gray-900 mb-2">Best Model</h4>
            <p className="text-2xl font-bold text-primary-600">GIN</p>
            <p className="text-sm text-gray-600">R² = 0.985, MAE = 0.05</p>
          </div>
          <div className="bg-white rounded-lg p-4">
            <h4 className="font-medium text-sm text-gray-900 mb-2">Fastest Model</h4>
            <p className="text-2xl font-bold text-green-600">RandomForest</p>
            <p className="text-sm text-gray-600">Avg time: 20ms</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AnalyticsTab
