/**
 * ConcentrationDashboard
 *
 * Renders time-resolved concentration profiles from a KMC simulation.
 * Each species is drawn as a coloured line; the 95 % Bayesian confidence
 * interval is rendered as a translucent shadow area beneath each curve.
 *
 * Props
 * -----
 * kmcResult  : KMCResponse   — full KMC simulation output
 * nodes      : RPGNode[]     — node metadata for labels and types
 * maxYieldAnnotation : boolean  — show/hide the max-yield time marker
 */

import React, { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Label,
} from 'recharts';
import type { KMCResponse, RPGNode } from '../lib/simulation-api';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PALETTE = [
  '#3b82f6', // blue
  '#10b981', // emerald
  '#f59e0b', // amber
  '#8b5cf6', // violet
  '#ef4444', // red
  '#06b6d4', // cyan
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Props {
  kmcResult: KMCResponse;
  nodes: RPGNode[];
  maxYieldAnnotation?: boolean;
}

interface ChartRow {
  time: number;
  [key: string]: number;   // species_id → concentration
}

// ---------------------------------------------------------------------------
// Helper: merge time + species data into Recharts-friendly row array
// ---------------------------------------------------------------------------

function buildChartRows(
  times: number[],
  speciesIds: string[],
  concentrations: Record<string, number[]>,
  lowerCI: Record<string, number[]>,
  upperCI: Record<string, number[]>,
): ChartRow[] {
  return times.map((t, i) => {
    const row: ChartRow = { time: t };
    for (const sid of speciesIds) {
      row[sid] = concentrations[sid]?.[i] ?? 0;
      row[`${sid}_lo`] = lowerCI[sid]?.[i] ?? 0;
      row[`${sid}_hi`] = upperCI[sid]?.[i] ?? 0;
    }
    return row;
  });
}

// ---------------------------------------------------------------------------
// Custom Tooltip
// ---------------------------------------------------------------------------

const CustomTooltip = ({
  active,
  payload,
  nodeMap,
}: {
  active?: boolean;
  payload?: any[];
  nodeMap: Record<string, RPGNode>;
}) => {
  if (!active || !payload?.length) return null;
  const t: number = payload[0]?.payload?.time ?? 0;
  const entries = payload.filter((p) => !p.dataKey.endsWith('_lo') && !p.dataKey.endsWith('_hi'));

  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-3 shadow-lg text-xs min-w-[160px]">
      <p className="font-semibold text-gray-600 dark:text-gray-300 mb-2">
        t = {t.toExponential(3)} s
      </p>
      {entries.map((p) => {
        const node = nodeMap[p.dataKey];
        return (
          <div key={p.dataKey} className="flex justify-between gap-4">
            <span style={{ color: p.color }}>{node?.label ?? p.dataKey}</span>
            <span className="font-mono text-gray-700 dark:text-gray-200">
              {(p.value as number).toExponential(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const ConcentrationDashboard: React.FC<Props> = ({
  kmcResult,
  nodes,
  maxYieldAnnotation = true,
}) => {
  const { times, species_ids, concentrations, lower_ci, upper_ci, max_yield_time, max_yield, n_trajectories } =
    kmcResult;

  const nodeMap = useMemo(
    () => Object.fromEntries(nodes.map((n) => [n.id, n])),
    [nodes]
  );

  const chartData = useMemo(
    () => buildChartRows(times, species_ids, concentrations, lower_ci, upper_ci),
    [times, species_ids, concentrations, lower_ci, upper_ci]
  );

  if (!chartData.length) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        Run a simulation to see concentration profiles.
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-800 dark:text-gray-100 text-base">
          Dynamic Concentration Dashboard
        </h3>
        <div className="flex gap-2 text-xs">
          <span className="bg-gray-100 dark:bg-gray-800 text-gray-500 px-2 py-1 rounded">
            {n_trajectories} trajectories
          </span>
          <span className="bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 px-2 py-1 rounded">
            95 % CI shaded
          </span>
        </div>
      </div>

      {/* Max yield annotation */}
      {maxYieldAnnotation && max_yield > 0 && (
        <div className="mb-3 flex items-center gap-2 text-xs bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-400 px-3 py-2 rounded-lg border border-emerald-200 dark:border-emerald-800">
          <span className="text-base">✓</span>
          <span>
            <strong>Max yield {(max_yield * 100).toFixed(1)} %</strong> at t ={' '}
            {max_yield_time.toExponential(3)} s
          </span>
        </div>
      )}

      {/* Chart */}
      <ResponsiveContainer width="100%" height={340}>
        <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

          <XAxis
            dataKey="time"
            tickFormatter={(v) => v.toExponential(1)}
            tick={{ fontSize: 11, fill: '#6b7280' }}
            type="number"
            scale="linear"
          >
            <Label
              value="Time (s)"
              offset={-10}
              position="insideBottom"
              style={{ fontSize: 12, fill: '#9ca3af' }}
            />
          </XAxis>

          <YAxis
            tickFormatter={(v) => v.toExponential(1)}
            tick={{ fontSize: 11, fill: '#6b7280' }}
            label={{
              value: 'Concentration (mol/L)',
              angle: -90,
              position: 'insideLeft',
              style: { fontSize: 12, fill: '#9ca3af' },
            }}
          />

          <Tooltip
            content={(props) => <CustomTooltip {...props} nodeMap={nodeMap} />}
          />
          <Legend
            formatter={(value) => nodeMap[value]?.label ?? value}
            wrapperStyle={{ fontSize: 12 }}
          />

          {/* Max-yield time vertical marker */}
          {maxYieldAnnotation && max_yield_time > 0 && (
            <ReferenceLine
              x={max_yield_time}
              stroke="#10b981"
              strokeDasharray="5 3"
              strokeWidth={2}
              label={{ value: 'max yield', fill: '#10b981', fontSize: 11 }}
            />
          )}

          {/* CI bands + mean lines per species */}
          {species_ids.map((sid, idx) => {
            const color = PALETTE[idx % PALETTE.length];
            return (
              <React.Fragment key={sid}>
                {/* Upper CI bound — defines the shaded ceiling */}
                <Area
                  type="monotone"
                  dataKey={`${sid}_hi`}
                  stroke="none"
                  fill={color}
                  fillOpacity={0.12}
                  legendType="none"
                  name={`${sid}_hi`}
                />
                {/* Lower CI bound — white-out to clip the shadow */}
                <Area
                  type="monotone"
                  dataKey={`${sid}_lo`}
                  stroke="none"
                  fill="white"
                  fillOpacity={0.8}
                  legendType="none"
                  name={`${sid}_lo`}
                />
                {/* Mean concentration line */}
                <Area
                  type="monotone"
                  dataKey={sid}
                  stroke={color}
                  strokeWidth={2}
                  fill="none"
                  dot={false}
                  name={sid}
                />
              </React.Fragment>
            );
          })}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ConcentrationDashboard;
