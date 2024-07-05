#include "vector.hpp"
#include "LSHIndex.hpp"
#include <stdexcept>
#include <cmath>
#include "serializer.hpp"
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
    m_lshIndex = std::make_unique<LSHIndex>(m_dimension, m_numberOfHashTables);
    m_lshIndex->buildIndex(m_vectors);
}

std::vector<size_t> VectorDatabase::findNearestNeighbors(const VectorXd& query, size_t k) const
{
    if (!m_lshIndex) {
        throw std::runtime_error("LSH index not initialized");
    }

    return m_lshIndex->findNearestNeighbors(query, k);
}

bool VectorDatabase::addVector(const VectorXd& vec)
{
    if (vec.size() != m_dimension) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    m_vectors.push_back(vec);
    m_idToIndexMap[m_nextId] = m_vectors.size() - 1;
    m_nextId++;

    // Update LSH index
    if (m_lshIndex) {
        m_lshIndex->addPoint(vec);
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
    return DatabaseSerializer::serialize(*this, filename);
}

bool VectorDatabase::deserialize(const std::string& filename)
{
    return DatabaseSerializer::deserialize(*this, filename);
}

void VectorDatabase::clear()
{
    m_vectors.clear();
    m_idToIndexMap.clear();
    m_nextId = 0;
    
    if (m_lshIndex) {m_lshIndex.reset();}
}