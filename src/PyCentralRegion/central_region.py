"""
central_region.py - Cyclotron Central Region Design Container

Container class for cyclotron central region designs.
Holds fields, RF cavities, geometry, and initial beam conditions.

Part of: PyCentralRegion module
Dependencies: PyPATools (field, species, particles)
"""

import numpy as np
import pickle
from typing import Optional, List, Dict, Union
from pathlib import Path
from PyPATools.field import Field
from PyPATools.species import IonSpecies
from PyPATools.particles import ParticleDistribution
from .rf_cavity import RFCavity

class CentralRegion:
    """
    Container for cyclotron central region design.

    Holds all components needed to simulate and optimize the central region:
    - Magnetic field (2D midplane or 3D)
    - Electric field (spiral inflector, etc.)
    - RF cavities
    - Geometry/boundaries
    - Ion species
    - Initial beam conditions

    Parameters
    ----------
    name : str
        Design name (default: "Cyclotron")
    dimensionality : str
        Tracking dimensionality: '2D' or '3D' (default: '2D')

    Attributes
    ----------
    name : str
        Design name
    dim : str
        '2D' or '3D'
    bfield : Field
        Magnetic field object
    efield : Field
        Electric field object (optional)
    rf_cavities : list
        List of RFCavity objects
    geometry : dict
        Geometry/boundary information
    species : IonSpecies
        Ion species
    metadata : dict
        Additional design information

    Examples
    --------
    > design = CentralRegion(name="CompactCyclotron", dimensionality='2D')
    > design.set_magnetic_field('midplane_field.dat')
    > design.set_species('H2_1+')
    > design.add_rf_cavity(cavity)
    > design.save('my_design.pkl')
    """

    def __init__(self, name: str = "Cyclotron", dimensionality: str = '2D'):

        if dimensionality not in ['2D', '3D']:
            raise ValueError("dimensionality must be '2D' or '3D'")

        self.name = name
        self.dim = dimensionality

        # Fields
        self.bfield = None
        self.efield = None

        # RF System
        self.rf_cavities = []

        # Geometry
        self.geometry = {
            'boundaries': None,
            'electrodes': None,
            'extraction_radius': None
        }

        # Beam
        self.species = None
        self.beam = None

        # Metadata
        self.metadata = {
            'description': '',
            'created': None,
            'modified': None,
            'author': '',
            'version': '1.0'
        }

    # ========================================================================
    # Field Management
    # ========================================================================

    def set_magnetic_field(self, field_or_filename: Union[object, str, Path], **kwargs):
        """
        Set magnetic field.

        Parameters
        ----------
        field_or_filename : Field, str, or Path
            Either a Field object or path to field file
        **kwargs
            Additional arguments passed to Field.from_file()
        """

        if isinstance(field_or_filename, Field):
            self.bfield = field_or_filename
        elif isinstance(field_or_filename, (str, Path)):
            self.bfield = Field.from_file(str(field_or_filename), **kwargs)
        else:
            raise TypeError("field_or_filename must be Field object or path to file")

        # Validate dimensionality
        if self.dim == '2D' and self.bfield.dim not in [0, 2]:
            print(f"Warning: 2D design with {self.bfield.dim}D field. "
                  f"Will only use midplane (z=0) values.")

        print(f"Magnetic field loaded: {self.bfield}")

    def set_electric_field(self, field_or_filename: Union[object, str, Path], **kwargs):
        """
        Set electric field (e.g., spiral inflector).

        Parameters
        ----------
        field_or_filename : Field, str, or Path
            Either a Field object or path to field file
        **kwargs
            Additional arguments passed to Field.from_file()
        """

        if isinstance(field_or_filename, Field):
            self.efield = field_or_filename
        elif isinstance(field_or_filename, (str, Path)):
            self.efield = Field.from_file(str(field_or_filename), **kwargs)
        else:
            raise TypeError("field_or_filename must be Field object or path to file")

        print(f"Electric field loaded: {self.efield}")

    # ========================================================================
    # RF Cavity Management
    # ========================================================================

    def add_rf_cavity(self, cavity):
        """
        Add RF cavity to design.

        Parameters
        ----------
        cavity : RFCavity
            RF cavity object
        """
        self.rf_cavities.append(cavity)
        print(f"Added RF cavity: {cavity}")

    def remove_rf_cavity(self, index: int):
        """Remove RF cavity by index."""
        if 0 <= index < len(self.rf_cavities):
            removed = self.rf_cavities.pop(index)
            print(f"Removed RF cavity: {removed}")
        else:
            raise IndexError(f"Cavity index {index} out of range")

    def clear_rf_cavities(self):
        """Remove all RF cavities."""
        self.rf_cavities = []
        print("Cleared all RF cavities")

    def set_bunch_phase(self, phase_deg: float):
        """
        Set global bunch phase offset for all RF cavities.

        Parameters
        ----------
        phase_deg : float
            Bunch phase offset [degrees]
        """
        for cavity in self.rf_cavities:
            cavity.set_bunch_phase_offset(phase_deg)

        if len(self.rf_cavities) > 0:
            print(f"Set bunch phase offset to {phase_deg:.2f} deg for {len(self.rf_cavities)} cavities")

    def set_rf_frequency(self, freq_hz: float):
        """
        Set RF frequency for all cavities.

        Parameters
        ----------
        freq_hz : float
            RF frequency [Hz]
        """
        for cavity in self.rf_cavities:
            cavity.set_frequency(freq_hz)

        if len(self.rf_cavities) > 0:
            print(f"Set RF frequency to {self.rf_cavities[0].harmonic * freq_hz / 1e6:.6f} MHz "
                  f"({freq_hz / 1e6:.6f} MHz base, harmonic {self.rf_cavities[0].harmonic}) for {len(self.rf_cavities)} cavities")

    # ========================================================================
    # Species & Beam
    # ========================================================================

    def set_species(self, species_or_name: Union[object, str]):
        """
        Set ion species.

        Parameters
        ----------
        species_or_name : IonSpecies or str
            Either IonSpecies object or species name ('proton', 'H2_1+', etc.)
        """

        if isinstance(species_or_name, IonSpecies):
            self.species = species_or_name
        elif isinstance(species_or_name, str):
            self.species = IonSpecies(species_or_name)
        else:
            raise TypeError("species_or_name must be IonSpecies object or string")

        self.beam = ParticleDistribution(species=self.species)

        print(f"Ion species set: {self.species.name}")

    def set_beam(self, beam):
        """
        Set initial beam distribution.

        Parameters
        ----------
        beam : ParticleDistribution
            Initial beam distribution
        """

        if not isinstance(beam, ParticleDistribution):
            raise TypeError("beam must be ParticleDistribution object")

        self.beam = beam
        self.species = beam.species

        print(f"Initial beam set: {beam.numpart} particles")

    # ========================================================================
    # Geometry
    # ========================================================================

    def set_boundaries(self, boundaries):
        """Set geometric boundaries (electrodes, walls, etc.)."""
        self.geometry['boundaries'] = boundaries

    def set_extraction_radius(self, radius: float):
        """
        Set extraction radius.

        Parameters
        ----------
        radius : float
            Extraction radius [m]
        """
        self.geometry['extraction_radius'] = radius

    # ========================================================================
    # Validation
    # ========================================================================

    def is_valid(self, verbose: bool = True) -> bool:
        """
        Check if design is valid for simulation.

        Parameters
        ----------
        verbose : bool
            Print validation messages

        Returns
        -------
        valid : bool
            True if design can be simulated
        """
        issues = []

        if self.bfield is None:
            issues.append("No magnetic field defined")

        if self.species is None:
            issues.append("No ion species defined")

        # 2D specific checks
        if self.dim == '2D':
            if self.bfield is not None and self.bfield.dim == 3:
                if verbose:
                    print("Info: 3D field will be evaluated at z=0 for 2D tracking")

        if issues:
            if verbose:
                print("Design validation failed:")
                for issue in issues:
                    print(f"  - {issue}")
            return False

        if verbose:
            print("Design validation passed")
        return True

    # ========================================================================
    # I/O
    # ========================================================================

    def to_dict(self) -> Dict:
        """
        Export design as dictionary.

        Returns
        -------
        design_dict : dict
            Dictionary representation of design
        """
        return {
            'name': self.name,
            'dimensionality': self.dim,
            'bfield': self.bfield,
            'efield': self.efield,
            'rf_cavities': self.rf_cavities,
            'geometry': self.geometry,
            'species': self.species,
            'beam': self.beam,
            'metadata': self.metadata
        }

    def save(self, filename: Union[str, Path]):
        """
        Save design to file.

        Parameters
        ----------
        filename : str or Path
            Output filename (.pkl extension recommended)
        """
        import datetime

        self.metadata['modified'] = datetime.datetime.now().isoformat()

        with open(filename, 'wb') as f:
            pickle.dump(self.to_dict(), f)

        print(f"Design saved to {filename}")

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'CentralRegion':
        """
        Load design from file.

        Parameters
        ----------
        filename : str or Path
            Input filename

        Returns
        -------
        design : CentralRegion
            Loaded design
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Create instance
        design = cls(name=data['name'], dimensionality=data['dimensionality'])

        # Restore attributes
        design.bfield = data.get('bfield')
        design.efield = data.get('efield')
        design.rf_cavities = data.get('rf_cavities', [])
        design.geometry = data.get('geometry', {})
        design.species = data.get('species')
        design.beam = data.get('beam')
        design.metadata = data.get('metadata', {})

        print(f"Design loaded from {filename}")
        return design

    # ========================================================================
    # Utility
    # ========================================================================

    def summary(self):
        """Print design summary."""
        print("=" * 70)
        print(f"CENTRAL REGION DESIGN: {self.name}")
        print("=" * 70)
        print(f"Dimensionality: {self.dim}")
        print(f"\nMagnetic Field: {self.bfield if self.bfield else 'Not set'}")
        print(f"Electric Field: {self.efield if self.efield else 'Not set'}")
        print(f"\nRF Cavities: {len(self.rf_cavities)}")
        for i, cavity in enumerate(self.rf_cavities):
            print(f"  {i}: {cavity.voltage / 1000:.1f} kV @ {cavity.frequency / 1e6:.1f} MHz")
        print(f"\nIon Species: {self.species.name if self.species else 'Not set'}")
        print(f"Initial Beam: {self.beam.numpart if self.beam else 0} particles")
        print(f"\nExtraction Radius: {self.geometry.get('extraction_radius', 'Not set')}")
        print("=" * 70)

    def __str__(self):
        return f"CentralRegion(name='{self.name}', dim='{self.dim}', " \
               f"species={self.species.name if self.species else None})"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    # Example usage
    print("Testing CentralRegion class...\n")

    # Create design
    design = CentralRegion(name="TestCyclotron", dimensionality='2D')

    # Set species
    design.set_species('H2_1+')

    # Set field (would normally load from file)
    from PyPATools.field import Field

    bfield = Field(dim=0, field={'x': 0, 'y': 0, 'z': 1.0}, label="Test 1T field")
    design.set_magnetic_field(bfield)

    # Add RF cavity
    cavity = RFCavity(p1=[0, -0.1, 0], p2=[0, 0.1, 0], voltage=60e3, frequency=168e6)
    design.add_rf_cavity(cavity)

    # Summary
    design.summary()

    # Validate
    design.is_valid()

    # Save/load
    design.save('test_design.pkl')
    design2 = CentralRegion.from_file('test_design.pkl')
    design2.summary()

    print("\nCentralRegion class working correctly!")
