"""Convert SMILES to molecular graphs for GNNs.

This module provides lightweight graph representations without RDKit dependency.
For production use, integrate RDKit for advanced features.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MolecularGraph:
    """Molecular graph representation.
    
    Attributes:
        node_features: Atom features (num_atoms, num_features)
        edge_index: Edge connectivity (2, num_edges)
        edge_features: Bond features (num_edges, num_features)
        num_nodes: Number of atoms
        num_edges: Number of bonds
    """
    
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: Optional[np.ndarray] = None
    num_nodes: int = 0
    num_edges: int = 0
    
    def __post_init__(self):
        if self.num_nodes == 0:
            self.num_nodes = len(self.node_features)
        if self.num_edges == 0:
            self.num_edges = self.edge_index.shape[1]


class SMILESToGraph:
    """Convert SMILES strings to molecular graphs.
    
    This is a simplified implementation without RDKit.
    For production, use RDKit for accurate molecular graphs.
    """
    
    # Atom feature vocabulary
    ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'H', 'OTHER']
    
    # Simple bond types
    BOND_TYPES = ['-', '=', '#', ':']
    
    def __init__(self):
        """Initialize SMILES to graph converter."""
        self.atom_to_idx = {atom: i for i, atom in enumerate(self.ATOM_TYPES)}
        self.bond_to_idx = {bond: i for i, bond in enumerate(self.BOND_TYPES)}
    
    def smiles_to_graph(self, smiles: str) -> MolecularGraph:
        """Convert SMILES to molecular graph.
        
        Args:
            smiles: SMILES string
        
        Returns:
            MolecularGraph object
        
        Note:
            This is a simplified implementation.
            For production, use RDKit: rdkit.Chem.MolFromSmiles(smiles)
        """
        # Parse atoms
        atoms = self._parse_atoms(smiles)
        
        # Create node features (one-hot encoding of atom types)
        node_features = self._create_node_features(atoms)
        
        # Create edge index (connectivity)
        edge_index = self._create_edge_index(len(atoms), smiles)
        
        # Create edge features (bond types)
        edge_features = self._create_edge_features(smiles, edge_index.shape[1])
        
        return MolecularGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features
        )
    
    def _parse_atoms(self, smiles: str) -> List[str]:
        """Extract atoms from SMILES string.
        
        Args:
            smiles: SMILES string
        
        Returns:
            List of atom symbols
        """
        atoms = []
        i = 0
        while i < len(smiles):
            # Skip non-atom characters
            if smiles[i] in '()[]=-#@/%\\' or smiles[i].isdigit():
                i += 1
                continue
            
            # Check for two-letter atoms (Cl, Br)
            if i + 1 < len(smiles) and smiles[i:i+2] in ['Cl', 'Br']:
                atoms.append(smiles[i:i+2])
                i += 2
            # Single-letter atoms
            elif smiles[i] in self.ATOM_TYPES:
                atoms.append(smiles[i])
                i += 1
            # Aromatic atoms (lowercase)
            elif smiles[i].lower() in [a.lower() for a in self.ATOM_TYPES]:
                atoms.append(smiles[i].upper())
                i += 1
            else:
                i += 1
        
        return atoms
    
    def _create_node_features(self, atoms: List[str]) -> np.ndarray:
        """Create one-hot encoded node features.
        
        Args:
            atoms: List of atom symbols
        
        Returns:
            Node feature matrix (num_atoms, num_atom_types)
        """
        num_atoms = len(atoms)
        num_features = len(self.ATOM_TYPES)
        
        node_features = np.zeros((num_atoms, num_features), dtype=np.float32)
        
        for i, atom in enumerate(atoms):
            idx = self.atom_to_idx.get(atom, self.atom_to_idx['OTHER'])
            node_features[i, idx] = 1.0
        
        return node_features
    
    def _create_edge_index(self, num_atoms: int, smiles: str) -> np.ndarray:
        """Create edge index (simplified connectivity).
        
        Args:
            num_atoms: Number of atoms
            smiles: SMILES string for bond information
        
        Returns:
            Edge index array (2, num_edges)
        
        Note:
            This creates a simple linear chain connectivity.
            For accurate graphs, use RDKit to parse SMILES properly.
        """
        edges = []
        
        # Simple linear chain (for demonstration)
        for i in range(num_atoms - 1):
            # Add edge in both directions (undirected graph)
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        
        if not edges:
            # Single atom molecule
            return np.zeros((2, 0), dtype=np.int64)
        
        edge_index = np.array(edges, dtype=np.int64).T
        return edge_index
    
    def _create_edge_features(self, smiles: str, num_edges: int) -> np.ndarray:
        """Create edge features (bond types).
        
        Args:
            smiles: SMILES string
            num_edges: Number of edges
        
        Returns:
            Edge feature matrix (num_edges, num_bond_types)
        """
        # Simplified: all single bonds
        num_features = len(self.BOND_TYPES)
        edge_features = np.zeros((num_edges, num_features), dtype=np.float32)
        edge_features[:, 0] = 1.0  # All single bonds
        
        return edge_features


class RDKitGraphConverter:
    """Convert molecules to graphs using RDKit (optional).
    
    This provides accurate molecular graph construction.
    Requires: pip install rdkit
    """
    
    def __init__(self):
        """Initialize RDKit converter."""
        try:
            from rdkit import Chem
            self.rdkit_available = True
            self.Chem = Chem
        except ImportError:
            self.rdkit_available = False
            print("RDKit not available. Install with: pip install rdkit")
    
    def smiles_to_graph(self, smiles: str) -> Optional[MolecularGraph]:
        """Convert SMILES to accurate molecular graph using RDKit.
        
        Args:
            smiles: SMILES string
        
        Returns:
            MolecularGraph object or None if RDKit unavailable
        """
        if not self.rdkit_available:
            raise ImportError("RDKit is required for accurate graph conversion")
        
        mol = self.Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Extract atom features
        node_features = self._get_atom_features(mol)
        
        # Extract bond connectivity
        edge_index, edge_features = self._get_bond_features(mol)
        
        return MolecularGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features
        )
    
    def _get_atom_features(self, mol) -> np.ndarray:
        """Extract atom features from RDKit molecule.
        
        Features:
            - Atom type (one-hot)
            - Degree
            - Formal charge
            - Hybridization
            - Aromaticity
        
        Args:
            mol: RDKit molecule object
        
        Returns:
            Atom feature matrix
        """
        features = []
        
        for atom in mol.GetAtoms():
            feature = [
                atom.GetAtomicNum(),  # Atomic number
                atom.GetDegree(),  # Number of bonds
                atom.GetFormalCharge(),  # Charge
                int(atom.GetIsAromatic()),  # Aromatic flag
                atom.GetTotalNumHs(),  # Number of hydrogens
            ]
            features.append(feature)
        
        return np.array(features, dtype=np.float32)
    
    def _get_bond_features(self, mol) -> Tuple[np.ndarray, np.ndarray]:
        """Extract bond features from RDKit molecule.
        
        Args:
            mol: RDKit molecule object
        
        Returns:
            Tuple of (edge_index, edge_features)
        """
        edges = []
        edge_attrs = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Undirected graph: add both directions
            edges.append([i, j])
            edges.append([j, i])
            
            # Bond features
            bond_type = bond.GetBondTypeAsDouble()  # 1.0, 1.5, 2.0, 3.0
            is_conjugated = int(bond.GetIsConjugated())
            is_in_ring = int(bond.IsInRing())
            
            bond_features = [bond_type, is_conjugated, is_in_ring]
            
            # Add features for both directions
            edge_attrs.append(bond_features)
            edge_attrs.append(bond_features)
        
        if not edges:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
        
        edge_index = np.array(edges, dtype=np.int64).T
        edge_features = np.array(edge_attrs, dtype=np.float32)
        
        return edge_index, edge_features
