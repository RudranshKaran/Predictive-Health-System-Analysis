import os
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import pprint

# Load environment variables
load_dotenv()

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
print(f"MongoDB URI: {mongo_uri}")

# Use explicit capitalized name instead of environment variable
correct_db_name = "Health_Summary"
print(f"Using database name: {correct_db_name}")

try:
    # Connect to MongoDB
    mongo_client = MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True
    )
    
    # List all available databases
    dbs = mongo_client.list_database_names()
    print(f"Available databases: {dbs}")
    
    # Select the database with correct capitalization
    db = mongo_client[correct_db_name]
    
    # List all collections in the database
    collections = db.list_collection_names()
    print(f"Available collections in {correct_db_name}: {collections}")
    
    if not collections:
        print(f"No collections found in {correct_db_name} database!")
        print("Let's check if there are collections in other databases...")
        for db_name in dbs:
            if db_name not in ['admin', 'local']:  # Skip system databases
                temp_db = mongo_client[db_name]
                temp_collections = temp_db.list_collection_names()
                print(f"Collections in {db_name}: {temp_collections}")
    
    # Define the patient ID to search for
    patient_id_str = "68185e32481c6023916a37db"
    print(f"\nLooking for patient ID: {patient_id_str}")
    
    # Try all collections to find the patient
    found = False
    for collection_name in collections:
        collection = db[collection_name]
        
        # Try with string ID
        patient = collection.find_one({"_id": patient_id_str})
        if patient:
            print(f"Found patient in {collection_name} as string ID")
            found = True
            pprint.pprint(patient)
            break
        
        # Try with ObjectId
        try:
            patient_id = ObjectId(patient_id_str)
            patient = collection.find_one({"_id": patient_id})
            if patient:
                print(f"Found patient in {collection_name} as ObjectId")
                found = True
                pprint.pprint(patient)
                break
        except Exception as e:
            print(f"Error converting to ObjectId: {str(e)}")
    
    if not found:
        print("Patient not found in any collection of the selected database")
        
        # Look in ALL databases including their collections
        print("\nSearching ALL databases for the patient ID...")
        for db_name in dbs:
            if db_name not in ['admin', 'local']:  # Skip system databases
                temp_db = mongo_client[db_name]
                temp_collections = temp_db.list_collection_names()
                
                for temp_collection_name in temp_collections:
                    temp_collection = temp_db[temp_collection_name]
                    
                    # Try with string ID
                    patient = temp_collection.find_one({"_id": patient_id_str})
                    if patient:
                        print(f"Found patient in database '{db_name}', collection '{temp_collection_name}' as string ID")
                        print("This is where your patient record is located!")
                        found = True
                        pprint.pprint(patient)
                        break
                    
                    # Try with ObjectId
                    try:
                        patient_id = ObjectId(patient_id_str)
                        patient = temp_collection.find_one({"_id": patient_id})
                        if patient:
                            print(f"Found patient in database '{db_name}', collection '{temp_collection_name}' as ObjectId")
                            print("This is where your patient record is located!")
                            found = True
                            pprint.pprint(patient)
                            break
                    except Exception as e:
                        pass  # Silently continue
                
                if found:
                    break
        
        if not found:
            print("Patient not found in any database or collection")
            
            # Show a sample document from collections across databases
            print("\nSample documents from collections:")
            for db_name in dbs:
                if db_name not in ['admin', 'local']:  # Skip system databases
                    temp_db = mongo_client[db_name]
                    temp_collections = temp_db.list_collection_names()
                    
                    for collection_name in temp_collections:
                        collection = temp_db[collection_name]
                        sample = list(collection.find().limit(1))
                        if sample:
                            print(f"\nSample from {db_name}.{collection_name}:")
                            print(f"ID type: {type(sample[0].get('_id'))}")
                            print(f"ID value: {sample[0].get('_id')}")
    
except Exception as e:
    print(f"MongoDB connection error: {str(e)}")