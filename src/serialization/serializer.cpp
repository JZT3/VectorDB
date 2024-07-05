#include "serializer.hpp"
#include "vector.hpp"
#include <fstream>
#include <stdexcept>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

bool DatabaseSerializer::serialize(const VectorDatabase& db, const std::string& filename)
{
    rapidjson::Document document;
    document.SetObject();
    rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

    // Serialize vectors
    rapidjson::Value vectorsArray(rapidjson::kArrayType);
    for (const auto& vec : db.getVectors()) {
        rapidjson::Value vectorObj(rapidjson::kArrayType);
        for (int i = 0; i < vec.size(); ++i) {
            vectorObj.PushBack(vec(i), allocator);
        }
        vectorsArray.PushBack(vectorObj, allocator);
    }
    document.AddMember("vectors", vectorsArray, allocator);

    // Serialize other properties
    document.AddMember("next_id", rapidjson::Value(db.getNextId()), allocator);
    document.AddMember("capacity", rapidjson::Value(db.getCapacity()), allocator);
    document.AddMember("dimension", rapidjson::Value(db.getDimension()), allocator);
    document.AddMember("distance_metric", rapidjson::Value(static_cast<int>(db.getDistanceMetric())), allocator);
    document.AddMember("storage_path", rapidjson::Value(db.getStoragePath().c_str(), allocator), allocator);
    document.AddMember("number_of_hash_tables", rapidjson::Value(db.getNumberOfHashTables()), allocator);
    document.AddMember("lsh_radius", rapidjson::Value(db.getLshRadius()), allocator);
    document.AddMember("mode", rapidjson::Value(static_cast<int>(db.getMode())), allocator);
    document.AddMember("log_level", rapidjson::Value(static_cast<int>(db.getLogLevel())), allocator);
    document.AddMember("index_type", rapidjson::Value(static_cast<int>(db.getIndexType())), allocator);

    // Convert to JSON string
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);

    // Write to file
    std::ofstream file(filename);
    if (!file) {
        return false;
    }

    file << buffer.GetString();
    return file.good();
}

bool DatabaseSerializer::deserialize(VectorDatabase& db, const std::string& filename)
{
    // Read file contents
    std::ifstream file(filename);
    if (!file) {
        return false;
    }

    std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Parse JSON
    rapidjson::Document document;
    if (document.Parse(json_str.c_str()).HasParseError()) {
        return false;
    }

    // Clear existing data
    db.clear();

    // Deserialize vectors
    const rapidjson::Value& vectorsArray = document["vectors"];
    for (const auto& vec_json : vectorsArray.GetArray()) {
        Eigen::VectorXd vec(vec_json.Size());
        for (rapidjson::SizeType i = 0; i < vec_json.Size(); ++i) {
            vec(i) = vec_json[i].GetDouble();
        }
        db.addVector(vec);
    }

    // Deserialize other properties
    db.setNextId(document["next_id"].GetUint64());
    db.setCapacity(document["capacity"].GetUint64());
    db.setDimension(document["dimension"].GetUint());
    db.setDistanceMetric(static_cast<VectorDatabase::DistanceMetric>(document["distance_metric"].GetInt()));
    db.setStoragePath(document["storage_path"].GetString());
    db.setNumberOfHashTables(document["number_of_hash_tables"].GetUint());
    db.setLshRadius(document["lsh_radius"].GetDouble());
    db.setMode(static_cast<VectorDatabase::Mode>(document["mode"].GetInt()));
    db.setLogLevel(static_cast<VectorDatabase::LogLevel>(document["log_level"].GetInt()));
    db.setIndexType(static_cast<VectorDatabase::IndexType>(document["index_type"].GetInt()));

    // Reinitialize components
    db.initializeDistanceFunction();
    db.initializeLSHIndex();

    return true;
}

std::string DatabaseSerializer::serializeVector(const Eigen::VectorXd& vec)
{
    // Implement vector serialization if needed
    return "";
}

Eigen::VectorXd DatabaseSerializer::deserializeVector(const std::string& data)
{
    // Implement vector deserialization if needed
    return Eigen::VectorXd();
}