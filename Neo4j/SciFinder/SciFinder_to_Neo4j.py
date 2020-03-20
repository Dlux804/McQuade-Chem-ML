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
        compound = Node("compound", reaction=raw_reactant, image_url=reactant_url)
        graph.merge(compound, 'compound', 'reaction')

    for product_url in products:
        raw_product = product_url
        product_url = reaction_dir + '/products/' + product_url
        product_url = 'http://localhost/' + product_url
        compound = Node("compound", reaction=raw_product, image_url=product_url)
        graph.merge(compound, 'compound', 'reaction')

    neo_reactants = Node("neo_reactants", reaction_ID=raw_reaction_dir)
    graph.merge(neo_reactants, "neo_reactants", "reaction_ID")

    neo_products = Node("neo_products", reaction_ID=raw_reaction_dir)
    graph.merge(neo_reactants, "neo_products", "reaction_ID")

    Rel = Relationship(neo_reactants, "produces", neo_products)
    tx = graph.begin()
    tx.create(Rel)
    tx.commit()

    matcher = NodeMatcher(graph)

    for reactant in reactants:
        reactant_node = matcher.match("compound", reaction=reactant).first()
        Rel = Relationship(reactant_node, "used_for", neo_reactants)
        tx = graph.begin()
        tx.create(Rel)
        tx.commit()

    for product in products:
        product_node = matcher.match("compound", reaction=product).first()
        Rel = Relationship(neo_products, "produces", product_node)
        tx = graph.begin()
        tx.create(Rel)
        tx.commit()