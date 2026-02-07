import { Thermometer, Droplets, TestTube2, Flame } from 'lucide-react'

interface ConditionsPanelProps {
  temperature: number
  pressure: number
  catalyst: string
  solvent: string
  onChange: (field: string, value: string | number) => void
}

const ConditionsPanel = ({
  temperature,
  pressure,
  catalyst,
  solvent,
  onChange,
}: ConditionsPanelProps) => {
  return (
    <div className="bg-white rounded-lg shadow">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900">Reaction Conditions</h3>
      </div>
      <div className="p-6 space-y-4">
        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
            <Thermometer className="w-4 h-4 mr-2 text-red-500" />
            Temperature (°C)
          </label>
          <input
            type="number"
            value={temperature}
            onChange={(e) => onChange('temperature', parseFloat(e.target.value))}
            min="-273.15"
            max="500"
            step="0.1"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
          <p className="mt-1 text-xs text-gray-500">
            Range: -273.15°C to 500°C
          </p>
        </div>

        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
            <Droplets className="w-4 h-4 mr-2 text-blue-500" />
            Pressure (atm)
          </label>
          <input
            type="number"
            value={pressure}
            onChange={(e) => onChange('pressure', parseFloat(e.target.value))}
            min="0"
            max="1000"
            step="0.1"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
          <p className="mt-1 text-xs text-gray-500">Range: 0 to 1000 atm</p>
        </div>

        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
            <Flame className="w-4 h-4 mr-2 text-orange-500" />
            Catalyst (optional)
          </label>
          <input
            type="text"
            value={catalyst}
            onChange={(e) => onChange('catalyst', e.target.value)}
            placeholder="e.g., H2SO4, Pd/C"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
        </div>

        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
            <TestTube2 className="w-4 h-4 mr-2 text-purple-500" />
            Solvent (optional)
          </label>
          <input
            type="text"
            value={solvent}
            onChange={(e) => onChange('solvent', e.target.value)}
            placeholder="e.g., water, THF, DCM"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
        </div>
      </div>
    </div>
  )
}

export default ConditionsPanel
