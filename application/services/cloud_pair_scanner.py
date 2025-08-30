# application/services/cloud_pair_scanner.py
"""Service zum Scannen und Erstellen von CloudPairs aus Dateisystem"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from domain.entities import CloudPair, ComparisonCase

logger = logging.getLogger(__name__)


@dataclass
class ScannerConfig:
    """Konfiguration für den CloudPair Scanner"""
    supported_formats: List[str] = None
    naming_conventions: Dict[str, str] = None

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['*.ply', '*.laz', '*.las', '*.xyz', '*.pts', '*.pcd']

        if self.naming_conventions is None:
            self.naming_conventions = {
                'ai_suffix': 'AI',
                'plain_suffix': 'plain',
                'separator': '-'
            }


class CloudPairScanner:
    """
    Service zum Scannen von Ordnern und Erstellen von CloudPairs.

    Diese Klasse kapselt die Logik zum:
    - Finden von Punktwolken-Dateien
    - Parsen von Dateinamen-Konventionen
    - Erstellen von CloudPair-Objekten
    - Bestimmen von Vergleichsfällen
    """

    def __init__(self, config: Optional[ScannerConfig] = None):
        """
        Initialisiert den Scanner mit Konfiguration.

        Args:
            config: Scanner-Konfiguration
        """
        self.config = config or ScannerConfig()
        self._init_comparison_cases()

    def _init_comparison_cases(self):
        """Initialisiert die Mapping-Tabelle für Vergleichsfälle"""
        self.comparison_cases = {
            ('a_ai', 'b_ai'): ComparisonCase.AI_VS_AI,
            ('a_ai', 'b_plain'): ComparisonCase.AI_VS_PLAIN,
            ('a_plain', 'b_ai'): ComparisonCase.PLAIN_VS_AI,
            ('a_plain', 'b_plain'): ComparisonCase.PLAIN_VS_PLAIN,
            # Umgekehrte Reihenfolgen
            ('b_ai', 'a_ai'): ComparisonCase.AI_VS_AI,
            ('b_plain', 'a_ai'): ComparisonCase.AI_VS_PLAIN,
            ('b_ai', 'a_plain'): ComparisonCase.PLAIN_VS_AI,
            ('b_plain', 'a_plain'): ComparisonCase.PLAIN_VS_PLAIN,
        }

    def scan_folder(
            self,
            folder_path: Path,
            indices: Optional[List[int]] = None
    ) -> List[CloudPair]:
        """
        Scannt einen Ordner nach CloudPairs.

        Args:
            folder_path: Pfad zum Ordner
            indices: Optionale Liste von Indizes zum Filtern

        Returns:
            Liste von CloudPair-Objekten
        """
        if not folder_path.exists():
            logger.warning(f"Folder does not exist: {folder_path}")
            return []

        logger.info(f"Scanning folder: {folder_path.name}")

        # Finde alle Punktwolken-Dateien
        point_cloud_files = self._find_point_cloud_files(folder_path)

        if not point_cloud_files:
            logger.warning(f"No point cloud files found in {folder_path}")
            logger.debug(f"Searched for formats: {self.config.supported_formats}")
            return []

        # Gruppiere Dateien nach Präfix
        groups = self._group_files_by_prefix(point_cloud_files, indices)

        # Erstelle CloudPairs
        cloud_pairs = self._create_cloud_pairs(groups, folder_path.name)

        return cloud_pairs

    def scan_multiple_folders(
            self,
            base_path: Path,
            folder_names: List[str],
            indices: Optional[List[int]] = None
    ) -> Dict[str, List[CloudPair]]:
        """
        Scannt mehrere Ordner und gibt CloudPairs gruppiert nach Ordner zurück.

        Args:
            base_path: Basis-Pfad
            folder_names: Liste von Ordnernamen
            indices: Optionale Index-Filter

        Returns:
            Dictionary mit Ordnername als Key und CloudPairs als Value
        """
        results = {}

        for folder_name in folder_names:
            folder_path = base_path / folder_name
            cloud_pairs = self.scan_folder(folder_path, indices)
            if cloud_pairs:
                results[folder_name] = cloud_pairs

        return results

    def _find_point_cloud_files(self, folder_path: Path) -> List[Path]:
        """Findet alle unterstützten Punktwolken-Dateien"""
        point_cloud_files = []

        for pattern in self.config.supported_formats:
            point_cloud_files.extend(folder_path.glob(pattern))

        logger.debug(f"Found {len(point_cloud_files)} point cloud files")
        return point_cloud_files

    def _parse_filename(self, file: Path) -> Optional[Tuple[str, int]]:
        """
        Parst einen Dateinamen und extrahiert Präfix und Index.

        Args:
            file: Dateipfad

        Returns:
            Tuple von (präfix, index) oder None wenn nicht parsbar
        """
        stem = file.stem  # z.B. "a-1", "a-1-AI", "b-1", "b-1-AI"
        parts = stem.split(self.config.naming_conventions['separator'])

        if len(parts) < 2:
            logger.debug(f"Cannot parse filename: {file.name}")
            return None

        # Format 1: a-1-AI oder b-1-AI (AI-enhanced)
        if parts[-1].upper() == self.config.naming_conventions['ai_suffix']:
            if len(parts) >= 3:
                prefix = f"{parts[0]}_ai"
                try:
                    index = int(parts[1])
                    return (prefix, index)
                except ValueError:
                    logger.debug(f"Non-numeric index in: {file.name}")
                    return None

        # Format 2: a_plain-1 oder a_ai-1 (alte Konvention)
        elif '_' in parts[0]:
            prefix = parts[0]
            try:
                index = int(parts[1])
                return (prefix, index)
            except ValueError:
                logger.debug(f"Non-numeric index in: {file.name}")
                return None

        # Format 3: a-1 oder b-1 (plain)
        else:
            prefix = f"{parts[0]}_plain"
            try:
                index = int(parts[1])
                return (prefix, index)
            except ValueError:
                logger.debug(f"Non-numeric index in: {file.name}")
                return None

        return None

    def _group_files_by_prefix(
            self,
            files: List[Path],
            indices: Optional[List[int]] = None
    ) -> Dict[str, List[Tuple[int, Path]]]:
        """Gruppiert Dateien nach ihrem Präfix"""
        groups = {}

        for file in files:
            parsed = self._parse_filename(file)
            if not parsed:
                continue

            prefix, index = parsed

            # Filter nach Index wenn angegeben
            if indices and index not in indices:
                continue

            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((index, file))

        # Log gefundene Gruppen
        for prefix, file_list in groups.items():
            indices_found = [idx for idx, _ in file_list]
            logger.info(f"  {prefix}: {len(file_list)} files (indices: {indices_found})")

        return groups

    def _create_cloud_pairs(
            self,
            groups: Dict[str, List[Tuple[int, Path]]],
            folder_id: str
    ) -> List[CloudPair]:
        """Erstellt CloudPairs aus gruppierten Dateien"""
        cloud_pairs = []
        prefixes = sorted(groups.keys())

        # Zähler für Vergleichsfälle
        case_counts = {case: 0 for case in ComparisonCase}

        # Erstelle Paare für alle Kombinationen
        for i, prefix1 in enumerate(prefixes):
            for prefix2 in prefixes[i + 1:]:
                pairs = self._create_pairs_for_prefixes(
                    prefix1, prefix2, groups, folder_id
                )

                for pair in pairs:
                    cloud_pairs.append(pair)
                    case_counts[pair.comparison_case] += 1

        # Log Zusammenfassung
        if cloud_pairs:
            logger.info(f"  Created {len(cloud_pairs)} cloud pairs:")
            for case, count in case_counts.items():
                if count > 0:
                    logger.info(f"    {case.value}: {count} pairs")

        return cloud_pairs

    def _create_pairs_for_prefixes(
            self,
            prefix1: str,
            prefix2: str,
            groups: Dict[str, List[Tuple[int, Path]]],
            folder_id: str
    ) -> List[CloudPair]:
        """Erstellt CloudPairs für zwei Präfixe"""
        pairs = []

        # Bestimme Vergleichsfall
        case_key = (prefix1, prefix2)
        if case_key not in self.comparison_cases:
            case_key = (prefix2, prefix1)

        if case_key not in self.comparison_cases:
            logger.debug(f"No comparison case for {prefix1} vs {prefix2}")
            return pairs

        comparison_case = self.comparison_cases[case_key]

        # Finde gemeinsame Indizes
        indices1 = {idx for idx, _ in groups[prefix1]}
        indices2 = {idx for idx, _ in groups[prefix2]}
        common_indices = indices1 & indices2

        # Erstelle Pairs für gemeinsame Indizes
        for idx in sorted(common_indices):
            file1 = next(f for i, f in groups[prefix1] if i == idx)
            file2 = next(f for i, f in groups[prefix2] if i == idx)

            # Bestimme mov/ref basierend auf Reihenfolge
            if case_key[0] == prefix1:
                mov_file, ref_file = file1, file2
            else:
                mov_file, ref_file = file2, file1

            cloud_pair = CloudPair(
                mov=str(mov_file.name),
                ref=str(ref_file.name),
                tag=f"{mov_file.stem}-{ref_file.stem}",
                folder_id=folder_id,
                comparison_case=comparison_case
            )
            pairs.append(cloud_pair)

        return pairs