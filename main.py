"""
Atomic Weights and Properties of the Elements are from:
National Institute of Standards and Technology (NIST),
URL: https://www.nist.gov/pml/atomic-data
"""

import os


ATOMIC_RADIUS = dict(
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


def parse_extxyz(file_content):
    frames = []
    lines = file_content.strip().splitlines()
    i = 0
    while i < len(lines):
        # Read the number of atoms
        num_atoms = int(lines[i].strip())
        i += 1

        # Read the frame"s metadata (assuming it"s a one-liner)
        metadata = lines[i].strip()
        i += 1

        atoms = []
        for _ in range(num_atoms):
            atom_line = lines[i].strip().split()
            atoms.append({
                "element": atom_line[0],
                "x": float(atom_line[1]),
                "y": float(atom_line[2]),
                "z": float(atom_line[3])
            })
            i += 1

        frames.append({"metadata": metadata, "atoms": atoms})

    return frames


def write_mol2(frames, output_file):
    with open(output_file, "w") as f:
        for frame_idx, frame in enumerate(frames):
            # Write molecule header
            f.write(f"@<TRIPOS>MOLECULE\n")
            f.write(f"Frame_{frame_idx + 1}\n")
            f.write(f"{len(frame["atoms"])} 0 0 0 0\n")
            f.write(f"SMALL\n")
            f.write(f"NO_CHARGES\n\n")

            # Write atom section
            f.write(f"@<TRIPOS>ATOM\n")
            for atom_idx, atom in enumerate(frame["atoms"]):
                f.write(
                    f"{atom_idx + 1} {atom["element"]}{atom_idx + 1} {atom["x"]:.4f} {atom["y"]:.4f} {atom["z"]:.4f} {atom["element"]} 1 LIG1 0.000\n")

            # Write bond section (empty as extxyz does not provide bond information)
            f.write(f"@<TRIPOS>BOND\n")

            f.write("\n")


def convert_extxyz_to_mol2(input_file, output_file):
    with open(input_file, "r") as f:
        file_content = f.read()

    frames = parse_extxyz(file_content)
    write_mol2(frames, output_file)


# Example usage
input_file = "input.extxyz"  # Replace with your extxyz file path
output_file = "output.mol2"  # Output MOL2 file path
convert_extxyz_to_mol2(input_file, output_file)
