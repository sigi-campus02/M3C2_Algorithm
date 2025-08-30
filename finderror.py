# Debug-Skript zum Prüfen der param_estimator.py
# Führen Sie dies in Ihrem Projektverzeichnis aus:

import sys
import os

# Füge Projektpfad hinzu
sys.path.insert(0, os.getcwd())

try:
    # Versuche die Datei zu importieren
    import application.services.param_estimator as pe

    print("✓ Modul erfolgreich importiert")
    print("\nVerfügbare Klassen und Funktionen:")
    print("-" * 40)

    # Liste alle verfügbaren Attribute
    for attr in dir(pe):
        if not attr.startswith('_'):
            obj = getattr(pe, attr)
            if isinstance(obj, type):
                print(f"  Klasse: {attr}")
            else:
                print(f"  {attr}: {type(obj).__name__}")

    # Prüfe spezifisch auf ParamEstimator
    print("\n" + "=" * 40)
    if hasattr(pe, 'ParamEstimator'):
        print("✓ ParamEstimator Klasse gefunden!")

        # Zeige Methoden der Klasse
        print("\nMethoden von ParamEstimator:")
        for method in dir(pe.ParamEstimator):
            if not method.startswith('_'):
                print(f"  - {method}")
    else:
        print("✗ ParamEstimator Klasse NICHT gefunden!")
        print("\nDateiinhalt (erste 100 Zeilen):")
        print("-" * 40)

        # Zeige Dateiinhalt
        filepath = os.path.join("application", "services", "param_estimator.py")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:100], 1):
                    print(f"{i:3}: {line.rstrip()}")
        else:
            print(f"Datei nicht gefunden: {filepath}")

except ImportError as e:
    print(f"✗ Import-Fehler: {e}")

except Exception as e:
    print(f"✗ Unerwarteter Fehler: {e}")

# Prüfe auch Python-Cache
print("\n" + "=" * 40)
print("Python Cache-Dateien:")
cache_dir = os.path.join("application", "services", "__pycache__")
if os.path.exists(cache_dir):
    for file in os.listdir(cache_dir):
        if "param_estimator" in file:
            filepath = os.path.join(cache_dir, file)
            mtime = os.path.getmtime(filepath)
            from datetime import datetime

            mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {file} - Geändert: {mod_time}")