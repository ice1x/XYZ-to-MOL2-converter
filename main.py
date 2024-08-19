import os


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
