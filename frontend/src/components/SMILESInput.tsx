import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { AlertCircle, CheckCircle2, X, Loader2 } from 'lucide-react'
import { validateSmiles } from '@/lib/api'

interface SMILESInputProps {
  label: string
  value: string
  onChange: (value: string) => void
  onRemove?: () => void
}

const SMILESInput = ({ label, value, onChange, onRemove }: SMILESInputProps) => {
  const [isFocused, setIsFocused] = useState(false)

  const validationMutation = useMutation({
    mutationFn: (smiles: string) => validateSmiles(smiles),
  })

  const handleBlur = () => {
    setIsFocused(false)
    if (value.trim()) {
      validationMutation.mutate(value)
    }
  }

  const showValidation = !isFocused && value.trim() !== ''
  const isValid = validationMutation.isSuccess && validationMutation.data?.is_valid
  const hasError = validationMutation.isSuccess && !validationMutation.data?.is_valid

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">{label}</label>
      <div className="relative">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={handleBlur}
          placeholder="e.g., CCO or c1ccccc1"
          className={`
            w-full px-3 py-2 pr-10 border rounded-md
            focus:outline-none focus:ring-2
            ${
              hasError
                ? 'border-red-300 focus:ring-red-500'
                : isValid
                ? 'border-green-300 focus:ring-green-500'
                : 'border-gray-300 focus:ring-primary-500'
            }
          `}
        />
        <div className="absolute inset-y-0 right-0 flex items-center pr-3 space-x-1">
          {validationMutation.isPending && (
            <Loader2 className="w-4 h-4 text-gray-400 animate-spin" />
          )}
          {showValidation && isValid && (
            <CheckCircle2 className="w-4 h-4 text-green-500" />
          )}
          {showValidation && hasError && (
            <AlertCircle className="w-4 h-4 text-red-500" />
          )}
          {onRemove && (
            <button
              onClick={onRemove}
              className="p-1 text-gray-400 hover:text-red-500 transition-colors"
              type="button"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
      {showValidation && isValid && validationMutation.data?.properties && (
        <div className="text-xs text-gray-500 space-y-1">
          <p>
            Atoms: {validationMutation.data.properties.num_atoms} |
            MW: {validationMutation.data.properties.estimated_mw.toFixed(1)} g/mol |
            Rings: {validationMutation.data.properties.num_rings}
            {validationMutation.data.properties.has_aromatic && ' | Aromatic'}
          </p>
        </div>
      )}
      {showValidation && hasError && (
        <p className="text-xs text-red-600">
          {validationMutation.data?.errors.join(', ') || 'Invalid SMILES'}
        </p>
      )}
    </div>
  )
}

export default SMILESInput
