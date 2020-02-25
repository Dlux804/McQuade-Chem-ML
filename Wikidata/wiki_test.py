from wikidata.client import Client

client = Client()
entity = client.get('Q20145', load=True)
print(entity.data)