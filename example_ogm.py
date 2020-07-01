"""
Objective: The goal of this script is to test py2neo's

"""

from py2neo import Graph, Relationship
from py2neo.ogm import GraphObject, Property, Label


class Person(GraphObject):  # Create Classes to make nodes for person
    name = Property()


class Car(GraphObject):
    name = Property()
    model = Property()
    asset = Label("Asset")


class House(GraphObject):
    name = Property()
    city = Property()
    asset = Label("Asset")


g = Graph("bolt://localhost:7687", user="neo4j", password="1234")
# Create Pete
p = Person()
p.name = "Pete"
g.push(p)

# Create Ferrari
c = Car()
c.name = "Ferrari"
c.asset = True
g.push(c)

# Create House
h = House()
h.name = "White House"
h.city = "New York"
h.asset = True
g.push(h)

# Drop down a level and grab the actual nodes
pn = p.__ogm__.node
cn = c.__ogm__.node

# Pete OWNS Ferrari (lower level py2neo)
ap = Relationship(pn, "OWNS", cn)
g.create(ap)

# Pete OWNS House (lower level py2neo)
hn = h.__ogm__.node
ah = Relationship(pn, "OWNS", hn)
g.create(ah)

# Grab & Print
# query = """MATCH (a:Person {name:'Pete'})-[:OWNS]->(n)
#            RETURN labels(n) as labels, n.name as name"""
# data = g.evaluate(query)
# for asset in data:
#     print(asset)
