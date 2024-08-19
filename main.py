import sys


def read_extxyz(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    atom_count = int(lines[0].strip())
    atoms = []

    for line in lines[2:2 + atom_count]:
        parts = line.split()
        atom_type = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append((atom_type, x, y, z))

    return atoms


def write_mol2(atoms, output_filename):
    with open(output_filename, 'w') as file:
        file.write("@<TRIPOS>MOLECULE\n")
        file.write("Converted Molecule\n")
        file.write(f"{len(atoms)} 0 0 0 0\n")
        file.write("SMALL\n")
        file.write("USER_CHARGES\n\n")

        file.write("@<TRIPOS>ATOM\n")
        for i, (atom_type, x, y, z) in enumerate(atoms, start=1):
            file.write(f"{i} {atom_type} {x:.4f} {y:.4f} {z:.4f} {atom_type} 1 <0> 0.0000\n")

        file.write("\n@<TRIPOS>BOND\n")
        # Bonds are not handled in this basic example
        file.write("\n")


def convert_extxyz_to_mol2(extxyz_file, mol2_file):
    atoms = read_extxyz(extxyz_file)
    write_mol2(atoms, mol2_file)


if __name__ == "__main__":
    # extxyz_file = "input.extxyz"  # Input file path
    extxyz_file = "train_mixedT10frames.xyz"
    mol2_file = "output.mol2"  # Output file path

    convert_extxyz_to_mol2(extxyz_file, mol2_file)
    print(f"Conversion completed: {mol2_file}")
