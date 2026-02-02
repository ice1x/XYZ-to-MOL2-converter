"""Convert extXYZ molecular structure files to MOL2 (Tripos Sybyl) format.

Supports multi-frame extXYZ files, automatic bond detection based on
covalent radii, and basic SYBYL atom type assignment.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import TextIO


# Covalent radii (in Angstroms) for bond detection.
#
# Source: National Institute of Standards and Technology (NIST),
#         "Atomic Weights and Fundamental Constants"
#         https://www.nist.gov/pml/atomic-data
#         Accessed: August 2024
#
# These are publicly available reference data and are not derived
# from any third-party software or repository.
ATOMIC_RADIUS: dict[str, float] = dict(
    Ac=1.88,
    Ag=1.59,
    Al=1.35,
    Am=1.51,
    As=1.21,
    Au=1.50,
    B=0.83,
    Ba=1.34,
    Be=0.35,
    Bi=1.54,
    Br=1.21,
    C=0.68,
    Ca=0.99,
    Cd=1.69,
    Ce=1.83,
    Cl=0.99,
    Co=1.33,
    Cr=1.35,
    Cs=1.67,
    Cu=1.52,
    D=0.23,
    Dy=1.75,
    Er=1.73,
    Eu=1.99,
    F=0.64,
    Fe=1.34,
    Ga=1.22,
    Gd=1.79,
    Ge=1.17,
    H=0.23,
    Hf=1.57,
    Hg=1.70,
    Ho=1.74,
    I=1.40,
    In=1.63,
    Ir=1.32,
    K=1.33,
    La=1.87,
    Li=0.68,
    Lu=1.72,
    Mg=1.10,
    Mn=1.35,
    Mo=1.47,
    N=0.68,
    Na=0.97,
    Nb=1.48,
    Nd=1.81,
    Ni=1.50,
    Np=1.55,
    O=0.68,
    Os=1.37,
    P=1.05,
    Pa=1.61,
    Pb=1.54,
    Pd=1.50,
    Pm=1.80,
    Po=1.68,
    Pr=1.82,
    Pt=1.50,
    Pu=1.53,
    Ra=1.90,
    Rb=1.47,
    Re=1.35,
    Rh=1.45,
    Ru=1.40,
    S=1.02,
    Sb=1.46,
    Sc=1.44,
    Se=1.22,
    Si=1.20,
    Sm=1.80,
    Sn=1.46,
    Sr=1.12,
    Ta=1.43,
    Tb=1.76,
    Tc=1.35,
    Te=1.47,
    Th=1.79,
    Ti=1.47,
    Tl=1.55,
    Tm=1.72,
    U=1.58,
    V=1.33,
    W=1.37,
    Y=1.78,
    Yb=1.94,
    Zn=1.45,
    Zr=1.56,
)

# Default tolerance factor for bond detection.
# A bond is detected when:  dist(A, B) < (r_cov_A + r_cov_B) * BOND_TOLERANCE
BOND_TOLERANCE: float = 1.2

# Basic SYBYL atom type mapping.
# Maps (element, number_of_bonds) -> SYBYL type.
# Falls back to element name if no specific mapping exists.
SYBYL_TYPES: dict[tuple[str, int], str] = {
    ("C", 4): "C.3",
    ("C", 3): "C.2",
    ("C", 2): "C.1",
    ("C", 1): "C.1",
    ("N", 3): "N.3",
    ("N", 2): "N.2",
    ("N", 1): "N.1",
    ("O", 2): "O.3",
    ("O", 1): "O.2",
    ("S", 2): "S.3",
    ("S", 1): "S.2",
    ("P", 4): "P.3",
    ("P", 3): "P.3",
    ("H", 1): "H",
    ("H", 0): "H",
    ("D", 1): "H",
    ("D", 0): "H",
    ("F", 1): "F",
    ("F", 0): "F",
    ("Cl", 1): "Cl",
    ("Cl", 0): "Cl",
    ("Br", 1): "Br",
    ("Br", 0): "Br",
    ("I", 1): "I",
    ("I", 0): "I",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Atom = dict[str, object]      # {"element": str, "x": float, "y": float, "z": float}
Bond = tuple[int, int]        # (atom_index_0, atom_index_1), 0-based
Frame = dict[str, object]     # {"metadata": str, "atoms": list[Atom], "bonds": list[Bond]}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_extxyz(file_content: str) -> list[Frame]:
    """Parse an extXYZ string into a list of frames.

    Each frame contains atom coordinates and optional metadata.
    The extXYZ format per frame is:
        <num_atoms>
        <metadata line>
        <element> <x> <y> <z>  (repeated num_atoms times)
    """
    frames: list[Frame] = []
    lines = file_content.strip().splitlines()
    total_lines = len(lines)
    i = 0

    while i < total_lines:
        # --- atom count ---
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            num_atoms = int(line)
        except ValueError:
            raise ValueError(
                f"Line {i + 1}: expected atom count (integer), got: {line!r}"
            )
        if num_atoms <= 0:
            raise ValueError(
                f"Line {i + 1}: atom count must be positive, got {num_atoms}"
            )
        i += 1

        # --- metadata ---
        if i >= total_lines:
            raise ValueError(
                f"Line {i + 1}: unexpected end of file (expected metadata line)"
            )
        metadata = lines[i].strip()
        i += 1

        # --- atoms ---
        atoms: list[Atom] = []
        for atom_no in range(num_atoms):
            if i >= total_lines:
                raise ValueError(
                    f"Line {i + 1}: unexpected end of file "
                    f"(expected atom {atom_no + 1}/{num_atoms})"
                )
            parts = lines[i].strip().split()
            if len(parts) < 4:
                raise ValueError(
                    f"Line {i + 1}: expected 'element x y z', got: {lines[i]!r}"
                )
            element = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                raise ValueError(
                    f"Line {i + 1}: cannot parse coordinates: {lines[i]!r}"
                )
            atoms.append({"element": element, "x": x, "y": y, "z": z})
            i += 1

        frames.append({"metadata": metadata, "atoms": atoms, "bonds": []})

    return frames


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------

def _distance(a: Atom, b: Atom) -> float:
    """Euclidean distance between two atoms."""
    dx = float(a["x"]) - float(b["x"])
    dy = float(a["y"]) - float(b["y"])
    dz = float(a["z"]) - float(b["z"])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def detect_bonds(
    atoms: list[Atom],
    tolerance: float = BOND_TOLERANCE,
) -> list[Bond]:
    """Detect bonds between atoms using covalent radii.

    Two atoms are considered bonded when:
        distance(A, B) < (cov_radius_A + cov_radius_B) * tolerance

    Atoms whose element is not in ATOMIC_RADIUS are skipped (no bonds).
    Returns a list of (i, j) pairs with i < j (0-based indices).
    """
    bonds: list[Bond] = []
    n = len(atoms)
    for i in range(n):
        elem_i = str(atoms[i]["element"])
        r_i = ATOMIC_RADIUS.get(elem_i)
        if r_i is None:
            continue
        for j in range(i + 1, n):
            elem_j = str(atoms[j]["element"])
            r_j = ATOMIC_RADIUS.get(elem_j)
            if r_j is None:
                continue
            max_dist = (r_i + r_j) * tolerance
            if _distance(atoms[i], atoms[j]) < max_dist:
                bonds.append((i, j))
    return bonds


def _get_sybyl_type(element: str, num_bonds: int) -> str:
    """Return a SYBYL atom type string for the given element and bond count."""
    return SYBYL_TYPES.get((element, num_bonds), element)


# ---------------------------------------------------------------------------
# MOL2 writing
# ---------------------------------------------------------------------------

def write_mol2(frames: list[Frame], output: TextIO) -> None:
    """Write frames to a file handle in MOL2 format.

    Each frame is written as a separate @<TRIPOS>MOLECULE block.
    Bond information and SYBYL types are included when available.
    """
    for frame_idx, frame in enumerate(frames):
        atoms: list[Atom] = frame["atoms"]  # type: ignore[assignment]
        bonds: list[Bond] = frame["bonds"]  # type: ignore[assignment]

        # Count bonds per atom for SYBYL type assignment
        bond_count: dict[int, int] = {}
        for a, b in bonds:
            bond_count[a] = bond_count.get(a, 0) + 1
            bond_count[b] = bond_count.get(b, 0) + 1

        # @<TRIPOS>MOLECULE
        output.write("@<TRIPOS>MOLECULE\n")
        output.write(f"Frame_{frame_idx + 1}\n")
        output.write(f"{len(atoms)} {len(bonds)} 0 0 0\n")
        output.write("SMALL\n")
        output.write("NO_CHARGES\n\n")

        # @<TRIPOS>ATOM
        output.write("@<TRIPOS>ATOM\n")
        for atom_idx, atom in enumerate(atoms):
            elem = str(atom["element"])
            name = f"{elem}{atom_idx + 1}"
            sybyl = _get_sybyl_type(elem, bond_count.get(atom_idx, 0))
            output.write(
                f"{atom_idx + 1:>7d} {name:<7s}"
                f" {float(atom['x']):>10.4f}"
                f" {float(atom['y']):>10.4f}"
                f" {float(atom['z']):>10.4f}"
                f" {sybyl:<6s} 1 LIG1 0.0000\n"
            )

        # @<TRIPOS>BOND
        output.write("@<TRIPOS>BOND\n")
        for bond_idx, (a, b) in enumerate(bonds):
            output.write(f"{bond_idx + 1:>6d} {a + 1:>4d} {b + 1:>4d} 1\n")

        output.write("\n")


# ---------------------------------------------------------------------------
# High-level conversion
# ---------------------------------------------------------------------------

def convert_extxyz_to_mol2(
    input_file: str,
    output_file: str,
    tolerance: float = BOND_TOLERANCE,
) -> None:
    """Read an extXYZ file, detect bonds, and write a MOL2 file."""
    with open(input_file, "r") as f:
        file_content = f.read()

    frames = parse_extxyz(file_content)

    for frame in frames:
        frame["bonds"] = detect_bonds(frame["atoms"], tolerance)  # type: ignore[arg-type]

    with open(output_file, "w") as f:
        write_mol2(frames, f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert extXYZ molecular files to MOL2 format.",
    )
    parser.add_argument("input", help="Path to the input extXYZ file")
    parser.add_argument("output", help="Path for the output MOL2 file")
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=BOND_TOLERANCE,
        help=(
            f"Bond detection tolerance factor (default: {BOND_TOLERANCE}). "
            "A bond is detected when dist < (r_a + r_b) * tolerance."
        ),
    )
    args = parser.parse_args(argv)

    try:
        convert_extxyz_to_mol2(args.input, args.output, args.tolerance)
    except FileNotFoundError:
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
