
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

#uri = "mongodb+srv://roberto:A7oUwYIroDOJXSr3@cluster0.yco7wow.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
uri = "mongodb+srv://frys:Ho2MfOfib1v4rXwE@cluster0.yco7wow.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)