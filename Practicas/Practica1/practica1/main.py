from practica1.analyze.correlation_analysis import CorrelationAnalysis
from practica1.analyze.dependency_analysis import DependencyAnalysis
from practica1.analyze.employee_distribution_analysis import EmployeeDistributionAnalysis
from practica1.analyze.faculty_vs_preparatory_analysis import FacultyVsPreparatoryAnalysis
from practica1.analyze.outlier_analysis import OutlierAnalysis
from practica1.analyze.salary_analisis_by_building import SalaryAnalysisByBuilding
from practica1.analyze.salary_equity_analysis import SalaryEquityAnalysis
from practica1.analyze.temporal_analysis import TemporalAnalysis


if __name__ == "__main__":
    file_csv = 'csv/typed_uanl.csv'
    analysis = [
        SalaryAnalysisByBuilding(file_csv),
        TemporalAnalysis(file_csv),
        DependencyAnalysis(file_csv),
        EmployeeDistributionAnalysis(file_csv),
        SalaryEquityAnalysis(file_csv),
        CorrelationAnalysis(file_csv),
        OutlierAnalysis(file_csv),
        FacultyVsPreparatoryAnalysis(file_csv)
    ]
    for row in analysis:
        row.analyze()