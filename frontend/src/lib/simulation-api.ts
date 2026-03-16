/**
 * Simulation API client for the KMC and RPG endpoints.
 *
 * Provides typed wrappers around POST /simulate/rpg and POST /simulate/kmc
 * so React components can call them with full TypeScript inference.
 */

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export interface RPGNode {
  id: string;
  smiles: string;
  type: 'reactant' | 'intermediate' | 'transition_state' | 'product';
  gibbs_free_energy: number;
  uncertainty: number;
  label: string;
}

export interface RPGEdge {
  id: string;
  from: string;
  to: string;
  activation_energy: number;
  frequency_factor: number;
  ea_uncertainty: number;
  is_rate_determining: boolean;
  rate_constant_at_T: number;
}

export interface RPGResponse {
  nodes: RPGNode[];
  edges: RPGEdge[];
  temperature: number;
  n_nodes: number;
  n_edges: number;
  rate_determining_step: string | null;
}

export interface KMCResponse {
  times: number[];
  species_ids: string[];
  concentrations: Record<string, number[]>;
  lower_ci: Record<string, number[]>;
  upper_ci: Record<string, number[]>;
  max_yield_time: number;
  max_yield: number;
  n_trajectories: number;
  rate_determining_step: string | null;
  graph: RPGResponse | null;
}

// ---------------------------------------------------------------------------
// Request builders
// ---------------------------------------------------------------------------

export interface RPGRequest {
  reactants: string[];
  products: string[];
  intermediates?: string[];
  activation_energies?: number[];
  frequency_factors?: number[];
  temperature?: number;
}

export interface KMCRequest extends RPGRequest {
  ea_uncertainties?: number[];
  initial_concentrations?: Record<string, number>;
  max_time?: number;
  n_snapshots?: number;
  n_trajectories?: number;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const simulationApi = {
  /** Build a Reaction Path Graph and return its node/edge structure. */
  buildRPG: (req: RPGRequest) => post<RPGResponse>('/simulate/rpg', req),

  /** Run a KMC simulation and return concentration profiles + CI. */
  runKMC: (req: KMCRequest) => post<KMCResponse>('/simulate/kmc', req),
};
