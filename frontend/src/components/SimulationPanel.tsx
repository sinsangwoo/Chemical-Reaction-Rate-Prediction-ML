/**
 * SimulationPanel
 *
 * Top-level panel that wires together:
 *   - Input form (reactants, products, intermediates, temperature)
 *   - ReactionPathGraph  (network visualisation)
 *   - EnergyProfileChart (Gibbs energy profile)
 *   - ConcentrationDashboard (KMC time-series)
 *
 * This component is designed to be dropped into the existing tab layout
 * alongside the current PredictionTab / AnalyticsTab components.
 */

import React, { useState } from 'react';
import { simulationApi, type KMCResponse, type RPGResponse } from '../lib/simulation-api';
import { ReactionPathGraph } from './ReactionPathGraph';
import { EnergyProfileChart } from './EnergyProfileChart';
import { ConcentrationDashboard } from './ConcentrationDashboard';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SimulationForm {
  reactants: string;
  products: string;
  intermediates: string;
  temperature: number;
  maxTime: number;
  nTrajectories: number;
}

const DEFAULT_FORM: SimulationForm = {
  reactants: 'CCO',
  products: 'CC=O',
  intermediates: 'CC[OH2+]',
  temperature: 500,
  maxTime: 1e-6,
  nTrajectories: 50,
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const SimulationPanel: React.FC = () => {
  const [form, setForm] = useState<SimulationForm>(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [kmc, setKmc] = useState<KMCResponse | null>(null);

  const rpg: RPGResponse | null = kmc?.graph ?? null;

  async function handleRun() {
    setLoading(true);
    setError(null);
    try {
      const result = await simulationApi.runKMC({
        reactants: form.reactants.split(',').map((s) => s.trim()).filter(Boolean),
        products: form.products.split(',').map((s) => s.trim()).filter(Boolean),
        intermediates: form.intermediates.split(',').map((s) => s.trim()).filter(Boolean),
        temperature: form.temperature,
        max_time: form.maxTime,
        n_trajectories: form.nTrajectories,
        n_snapshots: 200,
      });
      setKmc(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Input card */}
      <div className="bg-white dark:bg-gray-900 rounded-xl p-5 shadow">
        <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100 mb-4">
          Kinetic Simulation Setup
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Field
            label="Reactants (comma-separated SMILES)"
            value={form.reactants}
            onChange={(v) => setForm((f) => ({ ...f, reactants: v }))}
          />
          <Field
            label="Products"
            value={form.products}
            onChange={(v) => setForm((f) => ({ ...f, products: v }))}
          />
          <Field
            label="Intermediates (optional)"
            value={form.intermediates}
            onChange={(v) => setForm((f) => ({ ...f, intermediates: v }))}
          />
          <NumField
            label="Temperature (K)"
            value={form.temperature}
            onChange={(v) => setForm((f) => ({ ...f, temperature: v }))}
            min={100}
            max={2000}
          />
          <NumField
            label="Max simulation time (s)"
            value={form.maxTime}
            onChange={(v) => setForm((f) => ({ ...f, maxTime: v }))}
            min={1e-12}
            step={1e-7}
          />
          <NumField
            label="Gillespie trajectories"
            value={form.nTrajectories}
            onChange={(v) => setForm((f) => ({ ...f, nTrajectories: v }))}
            min={1}
            max={500}
            step={10}
          />
        </div>

        {error && (
          <div className="mt-3 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-3 py-2 rounded">
            {error}
          </div>
        )}

        <button
          onClick={handleRun}
          disabled={loading}
          className="mt-4 w-full md:w-auto px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium rounded-lg text-sm transition-colors"
        >
          {loading ? 'Simulating…' : 'Run KMC Simulation'}
        </button>
      </div>

      {/* Results */}
      {kmc && rpg && (
        <>
          <ReactionPathGraph
            nodes={rpg.nodes}
            edges={rpg.edges}
            rdStep={rpg.rate_determining_step}
          />
          <EnergyProfileChart
            nodes={rpg.nodes}
            edges={rpg.edges}
            rdStep={rpg.rate_determining_step}
            temperature={form.temperature}
          />
          <ConcentrationDashboard
            kmcResult={kmc}
            nodes={rpg.nodes}
            maxYieldAnnotation
          />
        </>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Mini helpers
// ---------------------------------------------------------------------------

const Field: React.FC<{ label: string; value: string; onChange: (v: string) => void }> = ({
  label,
  value,
  onChange,
}) => (
  <div>
    <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">{label}</label>
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2 text-sm bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
    />
  </div>
);

const NumField: React.FC<{
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}> = ({ label, value, onChange, min, max, step = 1 }) => (
  <div>
    <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">{label}</label>
    <input
      type="number"
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      min={min}
      max={max}
      step={step}
      className="w-full border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2 text-sm bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
    />
  </div>
);

export default SimulationPanel;
