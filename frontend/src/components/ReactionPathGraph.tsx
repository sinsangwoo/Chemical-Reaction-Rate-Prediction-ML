/**
 * ReactionPathGraph
 *
 * Renders an interactive node-link diagram of the elementary reaction
 * network built by the RPG module.  Built on SVG with a simple
 * force-simulation layout (no external library required).
 *
 * Colour coding:
 *   reactant        — blue
 *   intermediate    — purple
 *   transition_state — amber
 *   product         — green
 *
 * The rate-determining step edge is rendered with a red dashed stroke.
 *
 * Props
 * -----
 * nodes     : RPGNode[]
 * edges     : RPGEdge[]
 * rdStep    : string | null
 * width     : number   (default 680)
 * height    : number   (default 380)
 */

import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { RPGEdge, RPGNode } from '../lib/simulation-api';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NODE_COLOR: Record<string, string> = {
  reactant: '#3b82f6',
  intermediate: '#8b5cf6',
  transition_state: '#f59e0b',
  product: '#10b981',
};

const NODE_R = 26;

// ---------------------------------------------------------------------------
// Simple force-layout (no D3 dependency)
// ---------------------------------------------------------------------------

interface Position { x: number; y: number; }

function initialLayout(nodes: RPGNode[], width: number, height: number): Record<string, Position> {
  const pad = 80;
  const usable = width - pad * 2;
  const positions: Record<string, Position> = {};
  // Separate into layers by node type
  const layers: Array<RPGNode['type']> = ['reactant', 'intermediate', 'transition_state', 'product'];
  const grouped: Record<string, RPGNode[]> = {};
  for (const l of layers) grouped[l] = [];
  for (const n of nodes) {
    (grouped[n.type] ?? (grouped['intermediate'] ??= [])).push(n);
  }

  // Spread nodes by their index in the full list (left-to-right order)
  const step = nodes.length > 1 ? usable / (nodes.length - 1) : 0;
  nodes.forEach((node, i) => {
    const jitter = node.type === 'transition_state' ? -height * 0.18 : 0;
    positions[node.id] = {
      x: pad + i * step,
      y: height / 2 + jitter,
    };
  });
  return positions;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

const NodeCircle: React.FC<{
  node: RPGNode;
  pos: Position;
  selected: boolean;
  onClick: () => void;
}> = ({ node, pos, selected, onClick }) => {
  const color = NODE_COLOR[node.type] ?? '#6b7280';
  return (
    <g
      transform={`translate(${pos.x},${pos.y})`}
      style={{ cursor: 'pointer' }}
      onClick={onClick}
    >
      <circle
        r={NODE_R}
        fill={color}
        fillOpacity={selected ? 1 : 0.82}
        stroke={selected ? '#1e293b' : 'white'}
        strokeWidth={selected ? 3 : 2}
      />
      {node.type === 'transition_state' && (
        <polygon
          points="0,-14 12,8 -12,8"
          fill="white"
          fillOpacity={0.55}
        />
      )}
      <text
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={10}
        fontWeight={600}
        fill="white"
        style={{ pointerEvents: 'none', userSelect: 'none' }}
      >
        {node.label.length > 8 ? node.label.slice(0, 7) + '…' : node.label}
      </text>
      <text
        y={NODE_R + 12}
        textAnchor="middle"
        fontSize={9}
        fill="#6b7280"
        style={{ pointerEvents: 'none' }}
      >
        {node.gibbs_free_energy.toFixed(1)} kJ/mol
      </text>
    </g>
  );
};

const EdgeArrow: React.FC<{
  edge: RPGEdge;
  fromPos: Position;
  toPos: Position;
}> = ({ edge, fromPos, toPos }) => {
  const dx = toPos.x - fromPos.x;
  const dy = toPos.y - fromPos.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len === 0) return null;
  const ux = dx / len;
  const uy = dy / len;
  // Start and end adjusted for circle radius
  const x1 = fromPos.x + ux * NODE_R;
  const y1 = fromPos.y + uy * NODE_R;
  const x2 = toPos.x - ux * (NODE_R + 6);
  const y2 = toPos.y - uy * (NODE_R + 6);

  const isRDS = edge.is_rate_determining;
  const stroke = isRDS ? '#ef4444' : '#94a3b8';

  // Midpoint for Ea label
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2 - 10;

  return (
    <g>
      <defs>
        <marker
          id={`arrow-${edge.id}`}
          viewBox="0 0 10 10"
          refX={8}
          refY={5}
          markerWidth={6}
          markerHeight={6}
          orient="auto-start-reverse"
        >
          <path d="M2 1L8 5L2 9" fill="none" stroke={stroke} strokeWidth={1.5}
            strokeLinecap="round" strokeLinejoin="round" />
        </marker>
      </defs>
      <line
        x1={x1} y1={y1} x2={x2} y2={y2}
        stroke={stroke}
        strokeWidth={isRDS ? 2.5 : 1.5}
        strokeDasharray={isRDS ? '6 3' : undefined}
        markerEnd={`url(#arrow-${edge.id})`}
      />
      <text
        x={mx} y={my}
        textAnchor="middle"
        fontSize={9}
        fill={isRDS ? '#ef4444' : '#94a3b8'}
        fontWeight={isRDS ? 700 : 400}
      >
        {edge.activation_energy.toFixed(1)} kJ/mol
      </text>
    </g>
  );
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface Props {
  nodes: RPGNode[];
  edges: RPGEdge[];
  rdStep: string | null;
  width?: number;
  height?: number;
}

export const ReactionPathGraph: React.FC<Props> = ({
  nodes,
  edges,
  rdStep,
  width = 680,
  height = 360,
}) => {
  const [positions, setPositions] = useState<Record<string, Position>>({});
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    setPositions(initialLayout(nodes, width, height));
    setSelected(null);
  }, [nodes, width, height]);

  const selectedNode = selected ? nodes.find((n) => n.id === selected) : null;

  if (!nodes.length) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-400 text-sm">
        No reaction network to display.
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-800 dark:text-gray-100 text-base">
          Reaction Path Network
        </h3>
        <div className="flex gap-2">
          {Object.entries(NODE_COLOR).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1 text-xs text-gray-500">
              <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: color }} />
              {type.replace('_', ' ')}
            </div>
          ))}
        </div>
      </div>

      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        className="overflow-visible"
      >
        {/* Edges first (under nodes) */}
        {edges.map((edge) => {
          const from = positions[edge.from];
          const to = positions[edge.to];
          if (!from || !to) return null;
          return (
            <EdgeArrow key={edge.id} edge={edge} fromPos={from} toPos={to} />
          );
        })}

        {/* Nodes */}
        {nodes.map((node) => {
          const pos = positions[node.id];
          if (!pos) return null;
          return (
            <NodeCircle
              key={node.id}
              node={node}
              pos={pos}
              selected={selected === node.id}
              onClick={() => setSelected(selected === node.id ? null : node.id)}
            />
          );
        })}
      </svg>

      {/* Info panel for selected node */}
      {selectedNode && (
        <div className="mt-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg text-xs border border-gray-200 dark:border-gray-700">
          <p className="font-semibold text-gray-800 dark:text-gray-100 mb-1">
            {selectedNode.label}
          </p>
          <div className="grid grid-cols-2 gap-1 text-gray-500 dark:text-gray-400">
            <span>Type:</span>
            <span className="capitalize">{selectedNode.type.replace('_', ' ')}</span>
            <span>SMILES:</span>
            <span className="font-mono truncate">{selectedNode.smiles}</span>
            <span>ΔG:</span>
            <span>{selectedNode.gibbs_free_energy.toFixed(2)} kJ/mol</span>
            {selectedNode.uncertainty > 0 && (
              <>
                <span>Uncertainty:</span>
                <span>±{selectedNode.uncertainty.toFixed(2)} kJ/mol</span>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ReactionPathGraph;
