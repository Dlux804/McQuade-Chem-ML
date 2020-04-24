from Neo4j.US_patents.US_patents_xml_to_csv import US_grants_directory_to_csvs, clean_up_checker_files
from Neo4j.US_patents import bulk_insert

US_patents_directory = 'C:/Users/User/Desktop/5104873'  # Change this to location of directory
# US_grants_directory_to_csvs(US_patents_directory)
# clean_up_checker_files(US_patents_directory)
bulk_insert.US_patents_to_neo(US_patents_directory)
