#include "vector.hpp"
#include <stdexcept>
#include <cmath>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
#include <flann/flann.hpp>
#include <flann/algorithms/lsh_index.h>
#include <fstream>

// Private constructor used by Builder
VectorDatabase::VectorDatabase(const VectorBuilder& builder)
    : m_vectors(),
      m_idToIndexMap(),
      m_nextId(0),
      m_capacity(builder.initialCapacity),
      m_dimension(builder.dimension),
      m_distanceMetric(builder.distanceMetric),
      m_storagePath(builder.storagePath),
      m_numberOfHashTables(builder.numberOfHashTables),
      m_lshRadius(builder.lshRadius),
      //m_mode(builder.mode),
      m_logLevel(builder.logLevel),
      m_indexType(builder.indexType)
{
    m_vectors.reserve(m_capacity);
    initializeDistanceFunction();
    initializeLSHIndex();
}

VectorDatabase::VectorDatabase(VectorDatabase&& other) noexcept
    : m_vectors(std::move(other.m_vectors)),
      m_idToIndexMap(std::move(other.m_idToIndexMap)),
      m_nextId(other.m_nextId),
      m_capacity(other.m_capacity),
      m_dimension(other.m_dimension),
      m_distanceMetric(other.m_distanceMetric),
      m_storagePath(std::move(other.m_storagePath)),
      m_numberOfHashTables(other.m_numberOfHashTables),
      m_lshRadius(other.m_lshRadius),
      //m_mode(other.m_mode),
      m_logLevel(other.m_logLevel),
      m_indexType(other.m_indexType),
      m_distanceFunction(std::move(other.m_distanceFunction)),
      m_flannIndex(std::move(other.m_flannIndex)),
      m_flannDataset(std::move(other.m_flannDataset))
{
    // Reset other's state
    other.m_nextId = 0;
    other.m_capacity = 0;
}

VectorDatabase& VectorDatabase::operator=(VectorDatabase&& other) noexcept
{
    if (this != &other) {
        m_vectors = std::move(other.m_vectors);
        m_idToIndexMap = std::move(other.m_idToIndexMap);
        m_nextId = other.m_nextId;
        m_capacity = other.m_capacity;
        m_dimension = other.m_dimension;
        m_distanceMetric = other.m_distanceMetric;
        m_storagePath = std::move(other.m_storagePath);
        m_numberOfHashTables = other.m_numberOfHashTables;
        m_lshRadius = other.m_lshRadius;
        //m_mode = other.m_mode;
        m_logLevel = other.m_logLevel;
        m_indexType = other.m_indexType;
        m_distanceFunction = std::move(other.m_distanceFunction);
        m_flannIndex = std::move(other.m_flannIndex);
        m_flannDataset = std::move(other.m_flannDataset);
        
        // Reset other's state
        other.m_nextId = 0;
        other.m_capacity = 0;
    }
    return *this;
}

void VectorDatabase::initializeDistanceFunction()
{
    switch (m_distanceMetric) {
        // case DistanceMetric::EUCLIDEAN:
        //     m_distanceFunction = [](const VectorXd& a, const VectorXd& b) {
        //         return (a - b).norm();
        //     };
        //     break;
        // case DistanceMetric::MANHATTAN:
        //     m_distanceFunction = [](const VectorXd& a, const VectorXd& b) {
        //         return (a - b).lpNorm<1>();
        //     };
        //     break;
        case DistanceMetric::COSINE:
            m_distanceFunction = [](const VectorXd& a, const VectorXd& b) {
                double dot = a.dot(b);
                double norm_a = a.norm();
                double norm_b = b.norm();
                return 1.0 - (dot / (norm_a * norm_b));
            };
            break;
    }
}

void VectorDatabase::initializeLSHIndex()
{
    if (m_vectors.empty()) {return;} // No data to index

    // Convert vectors to FLANN-compatible format
    auto rawDataPtr = std::make_unique<double[]>(m_vectors.size() * m_dimension);
        m_flannDataset = std::make_shared<flann::Matrix<double>>(
            rawDataPtr.release(), m_vectors.size(), m_dimension);

    for (size_t i = 0; i < m_vectors.size(); ++i) {
        for (size_t j = i; j < m_dimension; ++j) {
            m_flannDataset[i][j] = m_vectors[i](j);
        }
    }

    // Set up LSH parameters
    flann::LshIndexParams indexParams;
    indexParams.table_number_ = m_numberOfHashTables;
    indexParams.key_size_ = 20;  // You may want to make this configurable
    indexParams.multi_probe_level_ = 2;  // You may want to make this configurable

    // Create and build the LSH index
    m_flannIndex = std::make_unique<flann::Index<flann::L2<double>>>(
        m_flannDataset, indexParams);
    m_flannIndex->buildIndex();
}

std::vector<size_t> VectorDatabase::findNearestNeighbors(const VectorXd& query, size_t k) const
{
    if (!m_flannIndex) {
        throw std::runtime_error("LSH index not initialized");
    }

    auto queryPoint = std::make_unique<flann::Matrix<double>>(
        new double[m_dimension], 1, m_dimension);
    
    for (size_t i = 0; i < m_dimension; ++i) {
        (*queryPoint)[0][i] = query(i);
    }

    std::vector<std::vector<int>> indices;
    std::vector<std::vector<double>> distances;
    flann::SearchParams searchParams(128);  // configurable
    m_flannIndex->knnSearch(*queryPoint, indices, distances, k, searchParams);

    // Convert FLANN indices to our internal indices
    std::vector<size_t> result;
    for (int index : indices[0]) {
        result.push_back(static_cast<size_t>(index))
    }

    return result;
}

bool VectorDatabase::addVector(const VectorXd& vec)
{
    if (vec.size() != m_dimension) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    m_vectors.push_back(vec);
    m_idToIndexMap[m_nextId] = m_vectors.size() - 1;
    m_nextId++;

    // Update FLANN index
    if (m_flannIndex) {
        flann::Matrix<double> newPoint(new double[m_dimension], 1, m_dimension);
        for (size_t i = 0; i < m_dimension; ++i) {
            newPoint[0][i] = vec(i);
        }

        m_flannIndex->addPoints(newPoint);
        delete[] newPoint.ptr();
    }

    return true;
}

VectorDatabase::VectorXd VectorDatabase::getVector(size_t index) const
{
    if (index >= m_vectors.size()) {
        throw std::out_of_range("Vector index out of range");
    }
    return m_vectors[index];
}

bool VectorDatabase::serialize(const std::string& filename) const
{
    rapidjson::Document document;
    document.SetObject();
    rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

    // Serialize vectors
    rapidjson::Value vectorsArray(rapidjson::kArrayType);
    for (const auto& vec : m_vectors) {
        rapidjson::Value vectorObj(rapidjson::kArrayType);
        for (int i = 0; i < vec.size(); ++i) {
            vectorObj.PushBack(vec(i), allocator);
        }
        vectorsArray.PushBack(vectorObj, allocator);
    }
    document.AddMember("vectors", vectorsArray, allocator);

    // Serialize other properties
    document.AddMember("next_id", rapidjson::Value(m_nextId), allocator);
    document.AddMember("capacity", rapidjson::Value(m_capacity), allocator);
    document.AddMember("dimension", rapidjson::Value(m_dimension), allocator);
    document.AddMember("distance_metric", rapidjson::Value(static_cast<int>(m_distanceMetric)), allocator);
    document.AddMember("storage_path", rapidjson::Value(m_storagePath.c_str(), allocator), allocator);
    document.AddMember("number_of_hash_tables", rapidjson::Value(m_numberOfHashTables), allocator);
    document.AddMember("lsh_radius", rapidjson::Value(m_lshRadius), allocator);
    //document.AddMember("mode", rapidjson::Value(static_cast<int>(m_mode)), allocator);
    document.AddMember("log_level", rapidjson::Value(static_cast<int>(m_logLevel)), allocator);
    document.AddMember("index_type", rapidjson::Value(static_cast<int>(m_indexType)), allocator);

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

bool VectorDatabase::deserialize(const std::string& filename)
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
    m_vectors.clear();
    m_idToIndexMap.clear();

    // Deserialize vectors
    const rapidjson::Value& vectorsArray = document["vectors"];
    for (const auto& vec_json : vectorsArray.GetArray()) {
        VectorXd vec(vec_json.Size());
        for (rapidjson::SizeType i = 0; i < vec_json.Size(); ++i) {
            vec(i) = vec_json[i].GetDouble();
        }
        m_vectors.push_back(std::move(vec));
    }
    
    // Deserialize other properties
    m_nextId = document["next_id"].GetUint64();
    m_capacity = document["capacity"].GetUint64();
    m_dimension = document["dimension"].GetUint();
    m_distanceMetric = static_cast<DistanceMetric>(document["distance_metric"].GetInt());
    m_storagePath = document["storage_path"].GetString();
    m_numberOfHashTables = document["number_of_hash_tables"].GetUint();
    m_lshRadius = document["lsh_radius"].GetDouble();
    //m_mode = static_cast<Mode>(document["mode"].GetInt());
    m_logLevel = static_cast<LogLevel>(document["log_level"].GetInt());
    m_indexType = static_cast<IndexType>(document["index_type"].GetInt());

    // Reinitialize components
    initializeDistanceFunction();
    initializeLSHIndex();

    return true;
}