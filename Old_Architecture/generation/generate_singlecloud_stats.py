from services.statistics_service import StatisticsService

folder_ids = ["data/TUNSPEKT Labordaten_all"]
filename_mov = "HandheldRoi"
filename_ref = "MavicRoi"
area_m2 = None
radius = 1.0
k = 6
sample_size = 100_000
use_convex_hull = True
out_path = "../outputs/TUNSPEKT_output/TUNSPEKT_m3c2_stats_clouds.xlsx"
sheet_name = "CloudStats"
output_format = "excel"

StatisticsService.calc_single_cloud_stats(
    folder_ids=folder_ids,
    filename_mov=filename_mov,
    filename_ref=filename_ref,
    area_m2=area_m2,
    radius=radius,
    k=k,
    sample_size=sample_size,
    use_convex_hull=use_convex_hull,
    out_path=out_path,
    sheet_name=sheet_name,
    output_format=output_format
)