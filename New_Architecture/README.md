# M3C2 Point Cloud Processing Pipeline - Refactored

## 📋 Übersicht

Dies ist eine vollständig refactorierte Version der M3C2 Point Cloud Processing Pipeline mit verbesserter Architektur, besserer Testbarkeit und klarer Trennung von Verantwortlichkeiten.

## 🏗️ Architektur

Das System folgt einer Clean Architecture mit folgenden Layern:

```
├── domain/              # Business Logic & Entities
│   ├── entities.py      # Domain-Objekte (CloudPair, M3C2Parameters, etc.)
│   ├── commands/        # Command Pattern für Pipeline-Schritte
│   ├── strategies/      # Strategy Pattern für Outlier Detection
│   ├── validators/      # Chain of Responsibility für Validierung
│   └── builders/        # Builder Pattern für Konfiguration
│
├── application/         # Use Cases & Orchestration
│   ├── orchestration/   # Pipeline-Orchestrierung
│   ├── factories/       # Factories mit Dependency Injection
│   └── services/        # Application Services
│
├── infrastructure/      # External Dependencies
│   └── repositories/    # Repository Pattern für Datenzugriff
│
├── presentation/        # UI/CLI/API (noch zu implementieren)
│
└── shared/             # Cross-cutting Concerns
    ├── config_loader.py # Konfiguration
    └── logging_setup.py # Logging
```

## 🚀 Installation

### 1. Virtual Environment erstellen

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows
```

### 2. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 3. Konfiguration anpassen

```bash
cp config_example.json config.json
# Edit config.json nach Bedarf
```

## 💻 Verwendung

### Basis-Verwendung

```bash
python main.py \
    --config config.json \
    --folders Multi-Illumination \
    --project MARS_Multi_Illumination
```

### Mit spezifischen Indizes

```bash
python main.py \
    --config config.json \
    --folders Multi-Illumination \
    --indices 1 2 3 \
    --project MARS_Part1
```

### Nur Statistiken berechnen

```bash
python main.py \
    --config config.json \
    --folders Multi-Illumination \
    --only-stats \
    --use-existing-params
```

### Mit Override-Parametern

```bash
python main.py \
    --config config.json \
    --folders Multi-Illumination \
    --normal-scale 0.002 \
    --search-scale 0.004
```

### Dry-Run Modus

```bash
python main.py \
    --config config.json \
    --folders Multi-Illumination \
    --dry-run
```

## 📁 Dateistruktur

### Eingabe-Struktur

```
data/
└── Multi-Illumination/
    ├── a-1.ply          # Original Part 1, Gruppe A
    ├── a-1-AI.ply       # AI-verarbeitet Part 1, Gruppe A
    ├── b-1.ply          # Original Part 1, Gruppe B
    ├── b-1-AI.ply       # AI-verarbeitet Part 1, Gruppe B
    └── ...
```

### Ausgabe-Struktur

```
outputs/
└── MARS_Multi_Illumination_output/
    ├── MARS_Multi_Illumination_statistics.xlsx
    ├── MARS_Multi_Illumination_plots/
    │   ├── part_1_histogram.png
    │   ├── part_1_colored.ply
    │   └── ...
    └── logs/
        └── orchestration.log
```

## 🔧 Konfiguration

### Wichtige Konfigurations-Parameter

| Parameter | Beschreibung | Default |
|-----------|--------------|---------|
| `data_path` | Basis-Verzeichnis für Daten | `data` |
| `output_path` | Ausgabe-Verzeichnis | `outputs` |
| `m3c2.normal_scale` | Normal-Radius für M3C2 | Auto-detect |
| `m3c2.search_scale` | Such-Radius für M3C2 | Auto-detect |
| `outlier_detection.method` | Methode für Outlier-Erkennung | `rmse` |
| `outlier_detection.multiplier` | Multiplikator für Threshold | `3.0` |
| `processing.sample_size` | Sample-Größe für Parameter-Schätzung | `10000` |
| `output.format` | Ausgabeformat für Statistiken | `excel` |

## 🧩 Design Patterns

### 1. **Repository Pattern**
- Abstrahiert Datenzugriff
- Ermöglicht einfaches Testen mit Mock-Repositories

### 2. **Command Pattern**
- Jeder Pipeline-Schritt ist ein Command
- Ermöglicht flexible Pipeline-Komposition

### 3. **Strategy Pattern**
- Verschiedene Outlier-Detection-Strategien
- Einfach erweiterbar mit neuen Methoden

### 4. **Builder Pattern**
- Fluent Interface für Pipeline-Konfiguration
- Validierung beim Build

### 5. **Factory Pattern**
- Service Factory mit Dependency Injection
- Pipeline Factory für Command-Komposition

### 6. **Chain of Responsibility**
- Validatoren können verkettet werden
- Jeder Validator prüft einen Aspekt

## 🧪 Testing

```bash
# Unit Tests
pytest tests/unit/

# Integration Tests
pytest tests/integration/

# Coverage Report
pytest --cov=. --cov-report=html
```

## 📊 Erweiterungen

### Neue Outlier-Detection-Methode hinzufügen

```python
# domain/strategies/outlier_detection.py
class MyOutlierStrategy(OutlierDetectionStrategy):
    def detect(self, distances: np.ndarray) -> Tuple[np.ndarray, float]:
        # Implementierung
        pass
```

### Neues Command hinzufügen

```python
# domain/commands/my_command.py
class MyCommand(Command):
    def execute(self, context: PipelineContext) -> PipelineContext:
        # Implementierung
        pass
    
    def can_execute(self, context: PipelineContext) -> bool:
        # Prüfung
        pass
```

## 🐛 Debugging

### Verbose Logging aktivieren

```bash
python main.py --log-level DEBUG
```

### Einzelne Konfiguration testen

```python
from application.factories.service_factory import ServiceFactory
from domain.builders.pipeline_builder import PipelineBuilder

# Setup
config = {...}
factory = ServiceFactory(config)

# Einzelne Konfiguration
builder = PipelineBuilder()
pipeline_config = builder.with_cloud_pair(...).build()

# Test
orchestrator = PipelineOrchestrator(factory)
result = orchestrator.run_single(pipeline_config)
```

## 📝 TODOs für Phase 2-5

- [ ] **Phase 2**: Repository Layer vollständig implementieren
- [ ] **Phase 3**: Alle Commands implementieren und testen
- [ ] **Phase 4**: Services modularisieren
- [ ] **Phase 5**: Umfassende Tests und Dokumentation

## 🤝 Migration vom alten Code

### Schritt 1: Backup erstellen
```bash
cp -r old_code old_code_backup
```

### Schritt 2: Schrittweise Migration
1. Neue Struktur parallel aufbauen
2. Services einzeln migrieren
3. Tests für jeden migrierten Service
4. Alte Komponenten nach erfolgreicher Migration entfernen

### Schritt 3: Validierung
- Ergebnisse mit altem System vergleichen
- Performance-Tests durchführen
- Edge Cases testen

## 📚 Weitere Dokumentation

- API-Dokumentation: `docs/api/`
- Architektur-Entscheidungen: `docs/adr/`
- Entwickler-Guide: `docs/developer-guide.md`

## 📄 Lizenz

[Deine Lizenz hier]