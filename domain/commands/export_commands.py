# New_Architecture/domain/commands/export_commands.py
"""Commands für verschiedene Export-Formate"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import pandas as pd
import numpy as np
from domain.commands.base import Command, PipelineContext

logger = logging.getLogger(__name__)


class ExportToExcelCommand(Command):
    """Exportiert alle Ergebnisse in eine Excel-Datei mit mehreren Sheets"""
    
    def __init__(self, export_service):
        super().__init__("ExportToExcel")
        self.export_service = export_service
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Exportiert zu Excel"""
        self.log_execution(context)
        
        config = context.get('config')
        output_path = Path(config.output_dir) / f"{config.project_name}_results.xlsx"
        
        # Erstelle Excel Writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Statistics Sheet
            if context.has('statistics_dataframe'):
                df_stats = context.get('statistics_dataframe')
                df_stats.to_excel(writer, sheet_name='Statistics', index=False)
                self._format_sheet(writer.sheets['Statistics'])
            
            # Distances Sheet
            if context.has('m3c2_distances'):
                distances = context.get('m3c2_distances')
                df_dist = self._create_distances_dataframe(distances)
                df_dist.to_excel(writer, sheet_name='Distances', index=False)
                self._format_sheet(writer.sheets['Distances'])
            
            # Parameters Sheet
            if context.has('m3c2_params'):
                params = context.get('m3c2_params')
                df_params = self._create_params_dataframe(params)
                df_params.to_excel(writer, sheet_name='Parameters', index=False)
                self._format_sheet(writer.sheets['Parameters'])
            
            # Cloud Info Sheet
            if context.has('cloud_statistics'):
                cloud_stats = context.get('cloud_statistics')
                df_cloud = self._create_cloud_stats_dataframe(cloud_stats)
                df_cloud.to_excel(writer, sheet_name='CloudInfo', index=False)
                self._format_sheet(writer.sheets['CloudInfo'])
            
            # Metadata Sheet
            df_meta = self._create_metadata_dataframe(context)
            df_meta.to_excel(writer, sheet_name='Metadata', index=False)
            self._format_sheet(writer.sheets['Metadata'])
        
        context.set('excel_export_path', output_path)
        logger.info(f"Exported results to Excel: {output_path}")
        
        return context
    
    def _create_distances_dataframe(self, distances: Dict) -> pd.DataFrame:
        """Erstellt DataFrame für Distanzen"""
        data = []
        
        if 'coordinates' in distances and 'with_outliers' in distances:
            coords = distances['coordinates']
            dists = distances['with_outliers']
            outliers = distances.get('outlier_mask', np.zeros(len(dists), dtype=bool))
            
            for i in range(len(dists)):
                data.append({
                    'X': coords[i, 0] if i < len(coords) else 0,
                    'Y': coords[i, 1] if i < len(coords) else 0,
                    'Z': coords[i, 2] if i < len(coords) else 0,
                    'Distance': dists[i],
                    'Is_Outlier': outliers[i] if i < len(outliers) else False
                })
        
        return pd.DataFrame(data)
    
    def _create_params_dataframe(self, params: Dict) -> pd.DataFrame:
        """Erstellt DataFrame für Parameter"""
        data = [{
            'Parameter': key,
            'Value': value
        } for key, value in params.items()]
        
        return pd.DataFrame(data)
    
    def _create_cloud_stats_dataframe(self, cloud_stats: Dict) -> pd.DataFrame:
        """Erstellt DataFrame für Cloud-Statistiken"""
        rows = []
        
        for cloud_type in ['moving_cloud', 'reference_cloud']:
            if cloud_type in cloud_stats:
                stats = cloud_stats[cloud_type]
                rows.append({
                    'Cloud': stats.get('label', cloud_type),
                    'Points': stats.get('point_count', 0),
                    'Volume': stats.get('volume', 0),
                    'Density': stats.get('estimated_density', 0),
                    'Avg_NN_Distance': stats.get('avg_nn_distance', 0)
                })
        
        return pd.DataFrame(rows)
    
    def _create_metadata_dataframe(self, context: PipelineContext) -> pd.DataFrame:
        """Erstellt DataFrame für Metadaten"""
        config = context.get('config')
        
        metadata = {
            'Property': ['Project', 'Cloud Pair', 'Processing Date', 'Pipeline Version'],
            'Value': [
                config.project_name,
                config.cloud_pair.tag,
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                '2.0.0'
            ]
        }
        
        # Füge Fehler hinzu, falls vorhanden
        if context.has_errors():
            errors = context.get_errors()
            metadata['Property'].append('Errors')
            metadata['Value'].append('; '.join(errors[:3]))  # Erste 3 Fehler
        
        return pd.DataFrame(metadata)
    
    def _format_sheet(self, worksheet) -> None:
        """Formatiert Excel-Sheet"""
        # Auto-fit Spaltenbreite
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob Excel-Export möglich ist"""
        return True  # Kann immer ausgeführt werden


class ExportToCloudCompareCommand(Command):
    """Exportiert Ergebnisse im CloudCompare-kompatiblen Format"""
    
    def __init__(self, export_service):
        super().__init__("ExportToCloudCompare")
        self.export_service = export_service
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Exportiert für CloudCompare"""
        self.log_execution(context)
        
        config = context.get('config')
        output_dir = Path(config.output_dir) / "cloudcompare"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Exportiere Punktwolke mit Distanzen als Scalar Field
        if context.has('cloud_ref') and context.has('m3c2_distances'):
            cloud = context.get('cloud_ref')
            distances = context.get('m3c2_distances')
            
            output_path = output_dir / f"{config.cloud_pair.tag}_with_distances.ply"
            self._export_cloud_with_scalars(cloud, distances, output_path)
            
            context.set('cloudcompare_export', output_path)
            logger.info(f"Exported CloudCompare file: {output_path}")
        
        # Exportiere CSV mit Distanzen
        if context.has('m3c2_distances'):
            csv_path = output_dir / f"{config.cloud_pair.tag}_distances.csv"
            self._export_distances_csv(context.get('m3c2_distances'), csv_path)
            logger.info(f"Exported distances CSV: {csv_path}")
        
        return context
    
    def _export_cloud_with_scalars(self, cloud: np.ndarray, distances: Dict, output_path: Path) -> None:
        """Exportiert Punktwolke mit Skalarfeldern"""
        import plyfile
        
        # Prepare data
        coords = distances.get('coordinates', cloud)
        dists = distances.get('with_outliers', np.zeros(len(coords)))
        outlier_mask = distances.get('outlier_mask', np.zeros(len(coords), dtype=bool))
        
        # Create structured array
        vertex = np.zeros(len(coords), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('distance', 'f4'), ('is_outlier', 'u1')
        ])
        
        vertex['x'] = coords[:, 0]
        vertex['y'] = coords[:, 1]
        vertex['z'] = coords[:, 2]
        vertex['distance'] = dists[:len(coords)]
        vertex['is_outlier'] = outlier_mask[:len(coords)]
        
        # Create PLY element
        el = plyfile.PlyElement.describe(vertex, 'vertex')
        plyfile.PlyData([el]).write(output_path)
    
    def _export_distances_csv(self, distances: Dict, output_path: Path) -> None:
        """Exportiert Distanzen als CSV"""
        df = pd.DataFrame({
            'Distance': distances.get('with_outliers', []),
            'Is_Outlier': distances.get('outlier_mask', [])
        })
        df.to_csv(output_path, index=False)
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Prüft ob CloudCompare-Export möglich ist"""
        return context.has('m3c2_distances')


class CreateReportCommand(Command):
    """Erstellt einen umfassenden HTML-Report"""
    
    def __init__(self, report_service):
        super().__init__("CreateReport")
        self.report_service = report_service
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Erstellt HTML-Report"""
        self.log_execution(context)
        
        config = context.get('config')
        output_path = Path(config.output_dir) / f"{config.project_name}_report.html"
        
        # Sammle alle Informationen für Report
        report_data = {
            'project': config.project_name,
            'cloud_pair': config.cloud_pair.tag,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': context.get('statistics'),
            'plots': context.get('plot_results', {}),
            'parameters': context.get('m3c2_params', {}),
            'errors': context.get_errors(),
            'execution_time': context.get('total_execution_time', 0)
        }
        
        # Generiere HTML
        html_content = self._generate_html_report(report_data)
        
        # Schreibe Report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        context.set('report_path', output_path)
        logger.info(f"Created HTML report: {output_path}")
        
        return context
    
    def _generate_html_report(self, data: Dict) -> str:
        """Generiert HTML-Report"""
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{data['project']} - M3C2 Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
                tr:nth-child(even) {{ background-color: #ecf0f1; }}
                .error {{ color: #e74c3c; }}
                .success {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .plot-container {{ margin: 20px 0; text-align: center; }}
                .plot-container img {{ max-width: 100%; height: auto; }}
                .metadata {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>M3C2 Analysis Report</h1>
            
            <div class="metadata">
                <h2>Metadata</h2>
                <p><strong>Project:</strong> {data['project']}</p>
                <p><strong>Cloud Pair:</strong> {data['cloud_pair']}</p>
                <p><strong>Generated:</strong> {data['timestamp']}</p>
                <p><strong>Processing Time:</strong> {data['execution_time']:.2f} seconds</p>
            </div>
            
            <h2>Parameters</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
        """
        
        # Add parameters
        for key, value in data['parameters'].items():
            html += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Statistics</h2>
        """
        
        # Add statistics if available
        if data['statistics']:
            stats = data['statistics']
            html += f"""
            <table>
                <tr>
                    <th>Metric</th>
                    <th>With Outliers</th>
                    <th>Inliers Only</th>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{stats.with_outliers.get('mean', 0):.4f}</td>
                    <td>{stats.inliers_only.get('mean', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Std Dev</td>
                    <td>{stats.with_outliers.get('std', 0):.4f}</td>
                    <td>{stats.inliers_only.get('std', 0):.4f}</td>
                </tr>
                <tr>
                    <td>RMSE</td>
                    <td>{stats.with_outliers.get('rmse', 0):.4f}</td>
                    <td>{stats.inliers_only.get('rmse', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Count</td>
                    <td>{stats.with_outliers.get('count', 0)}</td>
                    <td>{stats.inliers_only.get('count', 0)}</td>
                </tr>
            </table>
            <p><strong>Outlier Percentage:</strong> {stats.outlier_percentage:.1f}%</p>
            """
        
        # Add plots if available
        if data['plots']:
            html += "<h2>Visualizations</h2>"
            for plot_type, plot_path in data['plots'].items():
                if plot_path and Path(plot_path).exists():
                    html += f"""
                    <div class="plot-container">
                        <h3>{plot_type.replace('_', ' ').title()}</h3>
                        <img src="{plot_path}" alt="{plot_type}">
                    </div>
                    """
        
        # Add errors if any
        if data['errors']:
            html += """
            <h2>Errors and Warnings</h2>
            <ul class="error">
            """
            for error in data['errors']:
                html += f"<li>{error}</li>"
            html += "</ul>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Report kann immer erstellt werden"""
        return True


# Enhanced Pipeline Factory with new commands
class EnhancedPipelineFactory:
    """Erweiterte Pipeline Factory mit allen neuen Commands"""
    
    def __init__(self, service_factory):
        self.service_factory = service_factory
    
    def create_full_pipeline(self, config) -> List[Command]:
        """Erstellt vollständige Pipeline mit allen Features"""
        commands = []
        
        # 1. Load Point Clouds
        from domain.commands.m3c2_commands import LoadPointCloudsCommandV2
        commands.append(LoadPointCloudsCommandV2(
            self.service_factory.get_point_cloud_repository()
        ))
        
        # 2. Estimate Parameters (optional)
        if not config.use_existing_params:
            from domain.commands.m3c2_commands import EstimateParametersCommand
            commands.append(EstimateParametersCommand(
                self.service_factory.get_param_estimator()
            ))
        
        # 3. Run M3C2
        from domain.commands.m3c2_commands import RunM3C2Command
        commands.append(RunM3C2Command(
            self.service_factory.get_m3c2_runner()
        ))
        
        # 4. Detect Outliers
        from domain.commands.m3c2_commands import DetectOutliersCommand
        strategy = self.service_factory.get_outlier_strategy(
            config.outlier_detection_method
        )
        commands.append(DetectOutliersCommand(strategy))
        
        # 5. Calculate Statistics
        commands.append(CalculateStatisticsCommand(
            self.service_factory.get_statistics_service()
        ))
        
        # 6. Generate Plots
        plot_config = self.service_factory.get_config_value('plotting', {})
        commands.append(GeneratePlotsCommand(
            self.service_factory.get_plot_service(),
            plot_config
        ))
        
        # 7. Export Results
        export_format = config.output_format or 'excel'
        if export_format == 'excel':
            commands.append(ExportToExcelCommand(
                self.service_factory.get_export_service()
            ))
        
        # 8. CloudCompare Export (optional)
        if config.export_cloudcompare:
            commands.append(ExportToCloudCompareCommand(
                self.service_factory.get_export_service()
            ))
        
        # 9. Generate Report (optional)
        if config.generate_report:
            commands.append(CreateReportCommand(
                self.service_factory.get_report_service()
            ))
        
        # 10. Save Results
        from domain.commands.m3c2_commands import SaveResultsCommandV2
        commands.append(SaveResultsCommandV2(
            self.service_factory.get_distance_repository()
        ))
        
        return commands
    
    def create_statistics_only_pipeline(self, config) -> List[Command]:
        """Erstellt Pipeline nur für Statistik-Berechnung"""
        commands = []
        
        # 1. Load existing results
        from domain.commands.m3c2_commands import LoadCloudCompareResultsCommand
        commands.append(LoadCloudCompareResultsCommand(
            self.service_factory.get_distance_repository()
        ))
        
        # 2. Calculate Statistics
        commands.append(CalculateStatisticsCommand(
            self.service_factory.get_statistics_service()
        ))
        
        # 3. Generate Plots
        commands.append(GeneratePlotsCommand(
            self.service_factory.get_plot_service()
        ))
        
        # 4. Export
        commands.append(ExportToExcelCommand(
            self.service_factory.get_export_service()
        ))
        
        return commands
    
    def create_batch_pipeline(self, configs: List) -> List[Command]:
        """Erstellt Pipeline für Batch-Verarbeitung"""
        from domain.commands.base import CompositeCommand
        
        commands = []
        
        # Process each configuration
        for i, config in enumerate(configs):
            sub_commands = self.create_full_pipeline(config)
            composite = CompositeCommand(
                f"BatchJob_{i}_{config.cloud_pair.tag}",
                sub_commands
            )
            commands.append(composite)
        
        # Aggregate statistics
        commands.append(CalculateBatchStatisticsCommand(
            self.service_factory.get_statistics_service()
        ))
        
        # Generate batch plots
        commands.append(GenerateBatchPlotsCommand(
            self.service_factory.get_plot_service()
        ))
        
        # Export batch results
        commands.append(ExportToExcelCommand(
            self.service_factory.get_export_service()
        ))
        
        return commands