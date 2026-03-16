/**
 * EnergyProfileChart
 *
 * Renders the Gibbs free energy profile along the reaction coordinate.
 * Each species node is plotted as a horizontal shelf; transition states
 * appear as peaks.  The rate-determining step is highlighted with a red
 * dashed barrier annotation.
 *
 * Props
 * -----
 * nodes     : RPGNode[]  — ordered species/TS nodes from the RPG
 * edges     : RPGEdge[]  — elementary-step edges (for Ea labels)
 * rdStep    : string | null  — reaction_id of the rate-determining step
 * temperature : number  — display temperature in K
 */

import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Area,
  AreaChart,
  Label,
} from 'recharts';
import type { RPGEdge, RPGNode } from '../lib/simulation-api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Props {
  nodes: RPGNode[];
  edges: RPGEdge[];
  rdStep: string | null;
  temperature: number;
}

interface DataPoint {
  x: number;         // reaction coordinate (0, 0.5, 1, 1.5, …)
  energy: number;    // Gibbs free energy relative to reactant (kJ/mol)
  upper: number;     // energy + uncertainty (CI band)
  lower: number;     // energy − uncertainty (CI band)
  label: string;
  nodeType: string;
  isTS: boolean;
  isRDS: boolean;
}

// ---------------------------------------------------------------------------
// Helper: build a smooth chart data array from RPG nodes
// ---------------------------------------------------------------------------

function buildChartData(nodes: RPGNode[], edges: RPGEdge[], rdStep: string | null): DataPoint[] {
  return nodes.map((node, i) => {
    const relatedEdge = edges.find(
      (e) => e.from === node.id || e.to === node.id
    );
    const isRDS =
      node.type === 'transition_state' &&
      relatedEdge?.is_rate_determining === true;

    return {
      x: i * 0.5,
      energy: node.gibbs_free_energy,
      upper: node.gibbs_free_energy + node.uncertainty,
      lower: node.gibbs_free_energy - node.uncertainty,
      label: node.label,
      nodeType: node.type,
      isTS: node.type === 'transition_state',
      isRDS,
    };
  });
}

// ---------------------------------------------------------------------------
// Custom Tooltip
// ---------------------------------------------------------------------------

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d: DataPoint = payload[0].payload;
  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-3 shadow-lg text-sm">
      <p className="font-semibold text-gray-900 dark:text-gray-100">{d.label}</p>
      <p className="text-gray-500 dark:text-gray-400 capitalize">{d.nodeType.replace('_', ' ')}</p>
      <p className="text-blue-600 dark:text-blue-400">
        ΔG = <span className="font-mono">{d.energy.toFixed(1)} kJ/mol</span>
      </p>
      {d.uncertainty > 0 && (
        <p className="text-gray-400 text-xs">
          ±{(d.uncertainty).toFixed(1)} kJ/mol uncertainty
        </p>
      )}
      {d.isRDS && (
        <p className="text-red-500 font-semibold mt-1">⚠ Rate-determining step</p>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const NODE_COLORS: Record<string, string> = {
  reactant: '#3b82f6',
  intermediate: '#8b5cf6',
  transition_state: '#f59e0b',
  product: '#10b981',
};

export const EnergyProfileChart: React.FC<Props> = ({
  nodes,
  edges,
  rdStep,
  temperature,
}) => {
  const data = useMemo(() => buildChartData(nodes, edges, rdStep), [nodes, edges, rdStep]);

  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        No reaction data available
      </div>
    );
  }

  const rdsPoint = data.find((d) => d.isRDS);
  const minG = Math.min(...data.map((d) => d.lower)) - 10;
  const maxG = Math.max(...data.map((d) => d.upper)) + 10;

  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-800 dark:text-gray-100 text-base">
          Interactive Energy Profile
        </h3>
        <span className="text-xs text-gray-400 bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
          T = {temperature} K
        </span>
      </div>

      {rdsPoint && (
        <div className="mb-3 flex items-center gap-2 text-xs bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 px-3 py-2 rounded-lg border border-red-200 dark:border-red-800">
          <span className="text-base">⚠️</span>
          <span>
            <strong>Rate-determining step:</strong> {rdsPoint.label} — barrier at{' '}
            {rdsPoint.energy.toFixed(1)} kJ/mol
          </span>
        </div>
      )}

      <ResponsiveContainer width="100%" height={320}>
        <AreaChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
          <defs>
            <linearGradient id="ciGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.03} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: '#6b7280' }}
            angle={-30}
            textAnchor="end"
            height={50}
          >
            <Label
              value="Reaction coordinate"
              offset={-10}
              position="insideBottom"
              style={{ fontSize: 12, fill: '#9ca3af' }}
            />
          </XAxis>

          <YAxis
            domain={[minG, maxG]}
            tickFormatter={(v) => `${v.toFixed(0)}`}
            tick={{ fontSize: 11, fill: '#6b7280' }}
            label={{
              value: 'ΔG (kJ/mol)',
              angle: -90,
              position: 'insideLeft',
              style: { fontSize: 12, fill: '#9ca3af' },
            }}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* Confidence interval band */}
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="url(#ciGradient)"
            fillOpacity={1}
          />
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="white"
            fillOpacity={1}
          />

          {/* Main energy line */}
          <Line
            type="monotone"
            dataKey="energy"
            stroke="#3b82f6"
            strokeWidth={2.5}
            dot={({ cx, cy, payload }) => (
              <circle
                key={payload.label}
                cx={cx}
                cy={cy}
                r={payload.isTS ? 6 : 5}
                fill={NODE_COLORS[payload.nodeType] ?? '#6b7280'}
                stroke="white"
                strokeWidth={2}
              />
            )}
          />

          {/* Rate-determining step vertical marker */}
          {rdsPoint && (
            <ReferenceLine
              x={rdsPoint.label}
              stroke="#ef4444"
              strokeDasharray="5 3"
              strokeWidth={2}
              label={{ value: 'RDS', fill: '#ef4444', fontSize: 11 }}
            />
          )}
        </AreaChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 mt-2 justify-center">
        {Object.entries(NODE_COLORS).map(([type, color]) => (
          <div key={type} className="flex items-center gap-1 text-xs text-gray-500">
            <span
              className="inline-block w-3 h-3 rounded-full"
              style={{ background: color }}
            />
            {type.replace('_', ' ')}
          </div>
        ))}
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <span className="inline-block w-6 h-0.5 bg-red-400" style={{ borderTop: '2px dashed #ef4444' }} />
          rate-determining step
        </div>
      </div>
    </div>
  );
};

export default EnergyProfileChart;
