import os
from py2neo import Node, Relationship, NodeMatcher, Graph


graph = Graph()

for reaction_dir in os.listdir('reactions'):

    print(reaction_dir)

    raw_reaction_dir = reaction_dir
    reaction_dir = 'reactions/' + reaction_dir
    reactants = os.listdir(reaction_dir + '/reactants')
    products = os.listdir(reaction_dir + '/products')

    for reactant_url in reactants:
        raw_reactant = reactant_url
        reactant_url = reaction_dir + '/reactants/' + reactant_url
        reactant_url = 'http://localhost/' + reactant_url
        compound = Node("compound", compound_name=raw_reactant, image_url=reactant_url)
        graph.merge(compound, 'compound', 'compound_name')

    for product_url in products:
        raw_product = product_url
        product_url = reaction_dir + '/products/' + product_url
        product_url = 'http://localhost/' + product_url
        compound = Node("compound", compound_name=raw_product, image_url=product_url)
        graph.merge(compound, 'compound', 'compound_name')

    reaction = Node("reaction", reaction_ID=raw_reaction_dir)
    graph.merge(reaction, "reaction", "reaction_ID")

    matcher = NodeMatcher(graph)

    for reactant in reactants:
        reactant_node = matcher.match("compound", reaction=reactant).first()
        Rel = Relationship(reactant_node, "reactant", reaction)
        tx = graph.begin()
        tx.create(Rel)
        tx.commit()

    for product in products:
        product_node = matcher.match("compound", reaction=product).first()
        Rel = Relationship(reaction, "product", product_node)
        tx = graph.begin()
        tx.create(Rel)
        tx.commit()