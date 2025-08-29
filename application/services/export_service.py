# New_Architecture/application/services/export_service.py
"""Export Service für verschiedene Ausgabeformate"""

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExportService:
    """Service für Daten-Export in verschiedene Formate"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    def export_to_excel(
        self,
        data: Dict[str, pd.DataFrame],
        output_path: Path,
        metadata: Optional[Dict] = None
    ) -> None:
        """Exportiert Daten zu Excel mit mehreren Sheets"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write data sheets
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel limit: 31 chars
                
                # Auto-adjust columns width
                worksheet = writer.sheets[sheet_name[:31]]
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
            
            # Add metadata sheet if provided
            if metadata:
                df_meta = pd.DataFrame(list(metadata.items()), columns=['Property', 'Value'])
                df_meta.to_excel(writer, sheet_name='Metadata', index=False)
        
        logger.info(f"Exported data to Excel: {output_path}")
    
    def export_to_csv(
        self,
        data: pd.DataFrame,
        output_path: Path,
        **kwargs
    ) -> None:
        """Exportiert DataFrame zu CSV"""
        data.to_csv(output_path, index=False, **kwargs)
        logger.info(f"Exported data to CSV: {output_path}")
    
    def export_to_json(
        self,
        data: Dict[str, Any],
        output_path: Path,
        indent: int = 2
    ) -> None:
        """Exportiert Daten zu JSON"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        json_data = convert_numpy(data)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=indent)
        
        logger.info(f"Exported data to JSON: {output_path}")
    
    def export_to_html(
        self,
        data: Dict[str, Any],
        output_path: Path,
        template: Optional[str] = None
    ) -> None:
        """Exportiert Daten zu HTML"""
        if template:
            # Use custom template
            html_content = template.format(**data)
        else:
            # Generate basic HTML
            html_content = self._generate_basic_html(data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Exported data to HTML: {output_path}")
    
    def _generate_basic_html(self, data: Dict[str, Any]) -> str:
        """Generiert einfaches HTML aus Daten"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>M3C2 Export</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <h1>M3C2 Analysis Results</h1>
        """
        
        for key, value in data.items():
            html += f"<h2>{key}</h2>"
            
            if isinstance(value, pd.DataFrame):
                html += value.to_html(classes='data-table')
            elif isinstance(value, dict):
                html += "<table>"
                for k, v in value.items():
                    html += f"<tr><td>{k}</td><td>{v}</td></tr>"
                html += "</table>"
            else:
                html += f"<p>{value}</p>"
        
        html += """
        </body>
        </html>
        """
        
        return html