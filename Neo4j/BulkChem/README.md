##Propose of directory
Graphs databases have begun to become very popular among many
different industries. Although graph databases are powerful,
there has been very little use of them in a the chemistry space.
This directory attempts to merge chemistry and graph databases.
The main source of data that is being used is the bulk chemical
data (found in the bulkchem_datafiles), with plans to expand to
larger datasets, such as PubChem, if a working example can be
done with the bulkchem data. 

##Setting up Neo4j
The graph database UI of choice for this repo is Neo4j, as 
they have a lot of online support as well as many open source
packages in python that can interact with the Neo4j desktop.
The packages used in python to interact with Neo4j is py2neo. 
To use this directory, you need to first add data to the GB
of choice. This repo only works with the bulkchem data, so
the csv file containing the molecules must be present for this
to work. Also, you must have the neo4j desktop installed and
a database configured. When configuring the database, ensure
you set the password as 'password'. This serves as a default
and allows py2neo to interact with the database with needing a
password. Then make sure the database is running before 
using this.

##Inserting and relating the molecules
In this directory, you can find the main.py file. This file
is what the user modifies that calls the files to insert and
relate the molecules. The file is simple and only contains
the lines. 
<br />
init_neo_bulkchem(fragments_as_nodes=False)
create_relationships('strict_fragments')
<br />
The main permeter that needs to be changed is the
parameter inside of create_relationships. There are 3
different protocols right now 
<br />
'Rdkit_sim_score' : relate all molecules to other molecules
by a relationship "rdkit_sim_score", with an attribute of the
relationship as the actual value of the rdkit similarity score
<br />
'pka' : relate by same pka sites
'strict_fragments' : relate molecules that have the same
fragments
<br />
Please refer to ChemNeo4j_v4.py to be understand how
these functions were implemented. 