"""Tests for extXYZ to MOL2 converter."""

from __future__ import annotations

import io
import math
import pytest

from main import (
    ATOMIC_RADIUS,
    BOND_TOLERANCE,
    _distance,
    _get_sybyl_type,
    detect_bonds,
    parse_extxyz,
    write_mol2,
)


# ---------------------------------------------------------------------------
# Sample molecules in extXYZ format
# ---------------------------------------------------------------------------

# Water (H2O): O-H bond ~0.96 A, angle ~104.5 deg
WATER_XYZ = """\
3
Water molecule
O   0.0000  0.0000  0.1173
H   0.0000  0.7572 -0.4692
H   0.0000 -0.7572 -0.4692
"""

# Methane (CH4): C-H bond ~1.09 A, tetrahedral
METHANE_XYZ = """\
5
Methane molecule
C   0.0000  0.0000  0.0000
H   0.6276  0.6276  0.6276
H  -0.6276 -0.6276  0.6276
H  -0.6276  0.6276 -0.6276
H   0.6276 -0.6276 -0.6276
"""

# Two frames: H2 then He (single atom, no bonds)
MULTI_FRAME_XYZ = """\
2
H2 molecule
H   0.0000  0.0000  0.0000
H   0.0000  0.0000  0.7400
1
Single helium atom
He  0.0000  0.0000  0.0000
"""

# CO2 linear: O=C=O, C-O bond ~1.16 A
CO2_XYZ = """\
3
Carbon dioxide
C   0.0000  0.0000  0.0000
O   0.0000  0.0000  1.1600
O   0.0000  0.0000 -1.1600
"""

# Two distant atoms — should NOT be bonded
FAR_ATOMS_XYZ = """\
2
Two distant carbons
C   0.0000  0.0000  0.0000
C   0.0000  0.0000  10.0000
"""


# ---------------------------------------------------------------------------
# parse_extxyz
# ---------------------------------------------------------------------------

class TestParseExtxyz:
    def test_single_frame(self):
        frames = parse_extxyz(WATER_XYZ)
        assert len(frames) == 1
        assert len(frames[0]["atoms"]) == 3
        assert frames[0]["atoms"][0]["element"] == "O"
        assert frames[0]["metadata"] == "Water molecule"

    def test_multi_frame(self):
        frames = parse_extxyz(MULTI_FRAME_XYZ)
        assert len(frames) == 2
        assert len(frames[0]["atoms"]) == 2
        assert len(frames[1]["atoms"]) == 1
        assert frames[1]["atoms"][0]["element"] == "He"

    def test_coordinates_parsed_as_floats(self):
        frames = parse_extxyz(WATER_XYZ)
        o = frames[0]["atoms"][0]
        assert isinstance(o["x"], float)
        assert isinstance(o["y"], float)
        assert isinstance(o["z"], float)
        assert o["z"] == pytest.approx(0.1173, abs=1e-6)

    def test_empty_bonds_initially(self):
        frames = parse_extxyz(WATER_XYZ)
        assert frames[0]["bonds"] == []

    def test_invalid_atom_count(self):
        with pytest.raises(ValueError, match="expected atom count"):
            parse_extxyz("abc\nmetadata\n")

    def test_negative_atom_count(self):
        with pytest.raises(ValueError, match="must be positive"):
            parse_extxyz("-1\nmetadata\n")

    def test_truncated_atoms(self):
        truncated = "3\nmetadata\nC 0 0 0\nH 0 0 1\n"
        with pytest.raises(ValueError, match="unexpected end of file"):
            parse_extxyz(truncated)

    def test_bad_coordinates(self):
        bad = "1\nmetadata\nC x y z\n"
        with pytest.raises(ValueError, match="cannot parse coordinates"):
            parse_extxyz(bad)

    def test_short_atom_line(self):
        short = "1\nmetadata\nC 0 0\n"
        with pytest.raises(ValueError, match="expected 'element x y z'"):
            parse_extxyz(short)


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------

class TestDetectBonds:
    def test_water_bonds(self):
        """Water: O bonded to both H atoms, H atoms not bonded to each other."""
        frames = parse_extxyz(WATER_XYZ)
        bonds = detect_bonds(frames[0]["atoms"])
        # O-H bonds: indices (0,1) and (0,2)
        assert len(bonds) == 2
        assert (0, 1) in bonds
        assert (0, 2) in bonds

    def test_methane_bonds(self):
        """Methane: C bonded to 4 H atoms."""
        frames = parse_extxyz(METHANE_XYZ)
        bonds = detect_bonds(frames[0]["atoms"])
        assert len(bonds) == 4
        for i in range(1, 5):
            assert (0, i) in bonds

    def test_co2_bonds(self):
        """CO2: C bonded to both O atoms."""
        frames = parse_extxyz(CO2_XYZ)
        bonds = detect_bonds(frames[0]["atoms"])
        assert len(bonds) == 2
        assert (0, 1) in bonds
        assert (0, 2) in bonds

    def test_no_bond_for_distant_atoms(self):
        """Two carbons 10 A apart should not be bonded."""
        frames = parse_extxyz(FAR_ATOMS_XYZ)
        bonds = detect_bonds(frames[0]["atoms"])
        assert bonds == []

    def test_unknown_element_skipped(self):
        """Unknown elements should be skipped without error."""
        atoms = [
            {"element": "Xx", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "C", "x": 0.0, "y": 0.0, "z": 1.0},
        ]
        bonds = detect_bonds(atoms)
        assert bonds == []

    def test_tolerance_affects_detection(self):
        """Tight tolerance should find fewer bonds."""
        frames = parse_extxyz(WATER_XYZ)
        atoms = frames[0]["atoms"]
        bonds_strict = detect_bonds(atoms, tolerance=0.5)
        bonds_loose = detect_bonds(atoms, tolerance=2.0)
        assert len(bonds_strict) <= len(bonds_loose)

    def test_h2_needs_higher_tolerance(self):
        """H2 bond (0.74 A) is long relative to 2*r_cov_H (0.46 A).

        With default tolerance (1.2), max_dist = 0.552 < 0.74, so no bond.
        With tolerance ~1.7 it gets detected.
        """
        frames = parse_extxyz(MULTI_FRAME_XYZ)
        atoms = frames[0]["atoms"]
        assert detect_bonds(atoms, tolerance=BOND_TOLERANCE) == []
        assert len(detect_bonds(atoms, tolerance=1.7)) == 1


# ---------------------------------------------------------------------------
# Distance helper
# ---------------------------------------------------------------------------

class TestDistance:
    def test_same_point(self):
        a = {"element": "C", "x": 1.0, "y": 2.0, "z": 3.0}
        assert _distance(a, a) == pytest.approx(0.0)

    def test_unit_distance(self):
        a = {"element": "C", "x": 0.0, "y": 0.0, "z": 0.0}
        b = {"element": "C", "x": 1.0, "y": 0.0, "z": 0.0}
        assert _distance(a, b) == pytest.approx(1.0)

    def test_3d_distance(self):
        a = {"element": "C", "x": 1.0, "y": 2.0, "z": 3.0}
        b = {"element": "C", "x": 4.0, "y": 6.0, "z": 3.0}
        assert _distance(a, b) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# SYBYL types
# ---------------------------------------------------------------------------

class TestSybylTypes:
    def test_carbon_sp3(self):
        assert _get_sybyl_type("C", 4) == "C.3"

    def test_carbon_sp2(self):
        assert _get_sybyl_type("C", 3) == "C.2"

    def test_nitrogen_sp3(self):
        assert _get_sybyl_type("N", 3) == "N.3"

    def test_oxygen_sp3(self):
        assert _get_sybyl_type("O", 2) == "O.3"

    def test_hydrogen(self):
        assert _get_sybyl_type("H", 1) == "H"

    def test_unknown_falls_back_to_element(self):
        assert _get_sybyl_type("Fe", 6) == "Fe"


# ---------------------------------------------------------------------------
# MOL2 writing
# ---------------------------------------------------------------------------

class TestWriteMol2:
    def _write_and_get(self, frames):
        buf = io.StringIO()
        write_mol2(frames, buf)
        return buf.getvalue()

    def test_water_mol2_structure(self):
        frames = parse_extxyz(WATER_XYZ)
        frames[0]["bonds"] = detect_bonds(frames[0]["atoms"])
        text = self._write_and_get(frames)

        assert "@<TRIPOS>MOLECULE" in text
        assert "@<TRIPOS>ATOM" in text
        assert "@<TRIPOS>BOND" in text
        assert "Frame_1" in text

    def test_water_atom_count_in_header(self):
        frames = parse_extxyz(WATER_XYZ)
        frames[0]["bonds"] = detect_bonds(frames[0]["atoms"])
        text = self._write_and_get(frames)
        lines = text.strip().splitlines()
        # Third line of MOL2: "<num_atoms> <num_bonds> 0 0 0"
        count_line = lines[2]
        parts = count_line.split()
        assert parts[0] == "3"  # 3 atoms
        assert parts[1] == "2"  # 2 bonds (O-H, O-H)

    def test_bond_lines_present(self):
        frames = parse_extxyz(WATER_XYZ)
        frames[0]["bonds"] = detect_bonds(frames[0]["atoms"])
        text = self._write_and_get(frames)

        bond_section = text.split("@<TRIPOS>BOND\n")[1].strip()
        bond_lines = [l for l in bond_section.splitlines() if l.strip()]
        assert len(bond_lines) == 2

    def test_sybyl_types_in_output(self):
        frames = parse_extxyz(METHANE_XYZ)
        frames[0]["bonds"] = detect_bonds(frames[0]["atoms"])
        text = self._write_and_get(frames)
        assert "C.3" in text  # Carbon with 4 bonds = sp3

    def test_multi_frame_output(self):
        frames = parse_extxyz(MULTI_FRAME_XYZ)
        for f in frames:
            f["bonds"] = detect_bonds(f["atoms"])
        text = self._write_and_get(frames)

        assert text.count("@<TRIPOS>MOLECULE") == 2
        assert "Frame_1" in text
        assert "Frame_2" in text

    def test_no_bonds_for_isolated_atom(self):
        frames = parse_extxyz(MULTI_FRAME_XYZ)
        for f in frames:
            f["bonds"] = detect_bonds(f["atoms"])
        text = self._write_and_get(frames)

        # Second frame (He) should have 0 bonds
        molecules = text.split("@<TRIPOS>MOLECULE")
        he_block = molecules[2]  # index 0 is empty, 1 is H2, 2 is He
        assert "1 0 0 0 0" in he_block


# ---------------------------------------------------------------------------
# ATOMIC_RADIUS sanity checks
# ---------------------------------------------------------------------------

class TestAtomicRadius:
    def test_hydrogen_radius(self):
        assert ATOMIC_RADIUS["H"] == 0.23

    def test_carbon_radius(self):
        assert ATOMIC_RADIUS["C"] == 0.68

    def test_deuterium_matches_hydrogen(self):
        assert ATOMIC_RADIUS["D"] == ATOMIC_RADIUS["H"]

    def test_all_positive(self):
        for elem, r in ATOMIC_RADIUS.items():
            assert r > 0, f"{elem} has non-positive radius {r}"
