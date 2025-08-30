# application/orchestration/pipeline_orchestrator.py
"""Pipeline Orchestrator für Batch-Verarbeitung"""

import logging
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import traceback

from domain.commands.base import PipelineContext, Command
from domain.entities import PipelineConfiguration
from application.factories.service_factory import ServiceFactory

# Type hints only - vermeidet zirkulären Import
if TYPE_CHECKING:
    from application.factories.pipeline_factory import PipelineFactory

logger = logging.getLogger(__name__)


class Pipeline:
    """Ausführbare Pipeline mit Commands"""

    def __init__(self, commands: List[Command], name: str = "Pipeline"):
        self.commands = commands
        self.name = name
        self.execution_times: Dict[str, float] = {}

    def execute(self, initial_context: PipelineContext) -> PipelineContext:
        """
        Führt die Pipeline aus.

        Args:
            initial_context: Initialer Kontext

        Returns:
            Finaler Kontext nach Ausführung aller Commands
        """
        context = initial_context
        total_start = time.perf_counter()

        logger.info(f"Starting pipeline: {self.name}")
        logger.info(f"Commands to execute: {len(self.commands)}")

        for i, command in enumerate(self.commands, 1):
            command_start = time.perf_counter()

            try:
                # Prüfe Vorbedingungen
                if not command.can_execute(context):
                    error_msg = f"Cannot execute command {i}/{len(self.commands)}: {command.name}"
                    logger.error(error_msg)
                    context.add_error(error_msg)

                    # Entscheide ob Pipeline fortgesetzt werden soll
                    if self._is_critical_command(command):
                        logger.error("Critical command failed - stopping pipeline")
                        break
                    else:
                        logger.warning("Non-critical command failed - continuing")
                        continue

                # Führe Command aus
                logger.info(f"[{i}/{len(self.commands)}] Executing: {command.name}")
                context = command.execute(context)

                # Speichere Ausführungszeit
                elapsed = time.perf_counter() - command_start
                self.execution_times[command.name] = elapsed
                logger.info(f"[{i}/{len(self.commands)}] Completed: {command.name} ({elapsed:.2f}s)")

            except Exception as e:
                elapsed = time.perf_counter() - command_start
                error_msg = f"Command {command.name} failed after {elapsed:.2f}s: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                context.add_error(error_msg)

                if self._is_critical_command(command):
                    logger.error("Critical command failed - stopping pipeline")
                    break

        total_elapsed = time.perf_counter() - total_start
        logger.info(f"Pipeline completed in {total_elapsed:.2f}s")

        # Log Zusammenfassung
        self._log_summary(context)

        return context

    def _is_critical_command(self, command: Command) -> bool:
        """Bestimmt ob ein Command kritisch ist"""
        # Diese Commands sind kritisch - Pipeline stoppt bei Fehler
        critical_commands = [
            "LoadPointClouds",
            "RunM3C2",
            "SaveResults"
        ]
        return command.name in critical_commands

    def _log_summary(self, context: PipelineContext) -> None:
        """Loggt eine Zusammenfassung der Pipeline-Ausführung"""
        logger.info("=" * 60)
        logger.info("Pipeline Summary:")
        logger.info(f"  Total commands: {len(self.commands)}")
        logger.info(f"  Executed: {len(self.execution_times)}")
        logger.info(f"  Errors: {len(context.get_errors())}")

        if self.execution_times:
            logger.info("  Execution times:")
            for cmd, elapsed in self.execution_times.items():
                logger.info(f"    {cmd}: {elapsed:.2f}s")
            total = sum(self.execution_times.values())
            logger.info(f"  Total time: {total:.2f}s")

        if context.has_errors():
            logger.error("  Errors encountered:")
            for error in context.get_errors():
                logger.error(f"    - {error}")

        logger.info("=" * 60)


class PipelineOrchestrator:
    """Koordiniert die Ausführung mehrerer Pipelines"""

    def __init__(
            self,
            service_factory: ServiceFactory,
            pipeline_factory: Optional['PipelineFactory'] = None
    ):
        self.service_factory = service_factory
        self.pipeline_factory = pipeline_factory
        self.results: List[Dict[str, Any]] = []
        self.failed_configs: List[PipelineConfiguration] = []

        # Wenn keine Factory übergeben wurde, erstelle eine
        if self.pipeline_factory is None:
            from application.factories.pipeline_factory import PipelineFactory
            self.pipeline_factory = PipelineFactory(service_factory)

    def run_batch(
            self,
            configurations: List[PipelineConfiguration],
            parallel: bool = False,
            continue_on_error: bool = True
    ) -> None:
        """
        Führt Batch-Verarbeitung für mehrere Konfigurationen aus.

        Args:
            configurations: Liste von Pipeline-Konfigurationen
            parallel: Ob parallel verarbeitet werden soll (noch nicht implementiert)
            continue_on_error: Ob bei Fehlern fortgefahren werden soll
        """
        if not configurations:
            logger.warning("No configurations to process")
            return

        logger.info("=" * 80)
        logger.info(f"Starting batch processing: {len(configurations)} configurations")
        logger.info("=" * 80)

        if parallel:
            logger.warning("Parallel processing not yet implemented - falling back to sequential")

        # Sequenzielle Verarbeitung
        for i, config in enumerate(configurations, 1):
            logger.info("")
            logger.info(f"[{i}/{len(configurations)}] Processing: {config.cloud_pair.tag}")
            logger.info("-" * 60)

            try:
                result = self.run_single(config)
                self.results.append(result)

                if result.get('success', False):
                    logger.info(f"[{i}/{len(configurations)}] SUCCESS: {config.cloud_pair.tag}")
                else:
                    logger.warning(f"[{i}/{len(configurations)}] PARTIAL SUCCESS: {config.cloud_pair.tag}")
                    if not continue_on_error:
                        logger.error("Stopping batch due to error (continue_on_error=False)")
                        break

            except Exception as e:
                logger.error(f"[{i}/{len(configurations)}] FAILED: {config.cloud_pair.tag} - {str(e)}")
                self.failed_configs.append(config)

                if not continue_on_error:
                    logger.error("Stopping batch due to error (continue_on_error=False)")
                    break

        # Finaler Report
        self._generate_report()

    def run_single(self, config: PipelineConfiguration) -> Dict[str, Any]:
        """
        Führt eine einzelne Pipeline aus.

        Args:
            config: Pipeline-Konfiguration

        Returns:
            Dictionary mit Ergebnissen
        """
        start_time = time.perf_counter()

        # Erstelle Pipeline
        pipeline = self.pipeline_factory.create_pipeline(config)

        # Erstelle initialen Kontext
        context = self._create_initial_context(config)

        # Führe Pipeline aus
        final_context = pipeline.execute(context)

        # Extrahiere Ergebnisse
        elapsed = time.perf_counter() - start_time

        result = {
            'config': config,
            'success': not final_context.has_errors(),
            'execution_time': elapsed,
            'errors': final_context.get_errors(),
            'history': final_context.get_history()
        }

        # Füge Statistiken hinzu wenn vorhanden
        if final_context.has('statistics'):
            result['statistics'] = final_context.get('statistics')

        if final_context.has('m3c2_result'):
            m3c2_result = final_context.get('m3c2_result')
            result['m3c2_summary'] = {
                'valid_count': m3c2_result.valid_count,
                'nan_percentage': m3c2_result.nan_percentage,
                'parameters': m3c2_result.parameters_used.to_dict()
            }

        # Speichere aggregierte Statistiken wenn gewünscht
        if config.output_format in ['excel', 'json', 'csv']:
            self._save_statistics(result, config)

        return result

    def _create_initial_context(self, config: PipelineConfiguration) -> PipelineContext:
        """Erstellt den initialen Pipeline-Kontext"""
        context = PipelineContext()

        # Füge Konfiguration hinzu
        context.set('config', config)

        # Füge Service-Referenzen hinzu (optional)
        context.set('service_factory', self.service_factory)

        # Füge Zeitstempel hinzu
        import datetime
        context.set('timestamp', datetime.datetime.now().isoformat())

        return context

    def _save_statistics(self, result: Dict[str, Any], config: PipelineConfiguration) -> None:
        """Speichert Statistiken in gewünschtem Format"""
        if not result.get('statistics'):
            return

        try:
            stats_repo = self.service_factory.get_statistics_repository()

            # Konvertiere zu DataFrame
            import pandas as pd
            stats_df = pd.DataFrame([result['statistics']])

            # Bestimme Output-Pfad
            output_path = Path(config.output_base_path) / f"{config.project_name}_statistics.{config.output_format}"

            # Speichere basierend auf Format
            if config.output_format == 'excel':
                stats_repo.save_to_excel(stats_df, output_path)
            elif config.output_format == 'csv':
                stats_repo.save_to_csv(stats_df, output_path)
            elif config.output_format == 'json':
                stats_repo.save_to_json(result['statistics'], output_path)

            logger.info(f"Statistics saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")

    def _generate_report(self) -> None:
        """Generiert einen finalen Report"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 80)

        total = len(self.results) + len(self.failed_configs)
        successful = sum(1 for r in self.results if r.get('success', False))
        partial = len(self.results) - successful
        failed = len(self.failed_configs)

        logger.info(f"Total configurations: {total}")
        logger.info(f"  [OK] Successful: {successful}")
        if partial > 0:
            logger.info(f"  Partial success: {partial}")
        if failed > 0:
            logger.info(f"  [FAIL] Failed: {failed}")

        # Zeige Ausführungszeiten
        if self.results:
            times = [r['execution_time'] for r in self.results]
            logger.info(f"Execution times:")
            logger.info(f"  Average: {sum(times) / len(times):.2f}s")
            logger.info(f"  Min: {min(times):.2f}s")
            logger.info(f"  Max: {max(times):.2f}s")
            logger.info(f"  Total: {sum(times):.2f}s")

        # Zeige fehlgeschlagene Konfigurationen
        if self.failed_configs:
            logger.error("Failed configurations:")
            for cfg in self.failed_configs:
                logger.error(f"  - {cfg.cloud_pair.tag}")

        logger.info("=" * 80)

    def get_results(self) -> List[Dict[str, Any]]:
        """Gibt die Ergebnisse zurück"""
        return self.results

    def get_failed_configs(self) -> List[PipelineConfiguration]:
        """Gibt die fehlgeschlagenen Konfigurationen zurück"""
        return self.failed_configs

    def reset(self) -> None:
        """Setzt den Orchestrator zurück"""
        self.results.clear()
        self.failed_configs.clear()
        logger.info("Orchestrator reset")