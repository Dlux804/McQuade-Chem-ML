from Neo4j.US_patents.US_patents_xml_to_csv import US_grants_directory_to_csvs, clean_up_checker_files
from Neo4j.US_patents import bulk_insert

"""
This script is designed to insert the entire US Patent dataset into Neo4j. 

To use:
Make sure all directories are unzipped,
and the only two directories that should be in the main directory are Applications and Grants. For example, the path
to an example file should be (on windows), 'C:/Users/User/Desktop/5104873/grants/1976/pftaps19760106_wk01.xml' with the
US_patents_directory set to 'C:/Users/User/Desktop/5104873'

Info about stopping script:
When this script inserts all the files into Neo4j, there are times during the script when the user can stop the process 
and resume later, mainly during the most lengthy parts of the script. The user is prompted when it is safe to stop and 
when it is not safe to stop the script and resume later. 
    If during the event you have to pause or stop the script for whatever reason during a not safe to pause section of 
the script, you will likely have to delete temp_files directory and the graph on Neo4j that you were insertting into and 
restart the script from the start, why this is the case will be explained in the README.md file.

Difference between this script and the main.py:
This script is meant to import that entire dataset as efficiently as possible. Whereas the main.py is designed to
import a single csv files, or multiple csv files, this file is designed to import the entire dataset.
"""

US_patents_directory = 'C:/Users/User/Desktop/5104873'  # Change this to location of directory
# US_grants_directory_to_csvs(US_patents_directory)  # Create csv directories
# clean_up_checker_files(US_patents_directory)  # Clean up checker files
bulk_insert.US_patents_to_neo(US_patents_directory)  # Bulk Insert directory
