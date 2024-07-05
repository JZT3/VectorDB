#include "LSHIndex.hpp"
#include <stdexcept>
#include <flann/flann.hpp>
#include <flann/algorithms/lsh_index.h>

LSHIndex::LSHIndex(size_t dimension, size_t numberOfHashTables)
    : m_dimension(dimension),
      m_numberOfHashTables(numberOfHashTables),
      m_flannIndex(nullptr),
      m_flannDataset(nullptr)
{
}

void LSHIndex::buildIndex(const std::vector<VectorXd>& vectors)
{
    if (vectors.empty()) {
        return; // No data to index
    }

    // Convert vectors to FLANN-compatible format
    auto rawDataPtr = std::make_unique<double[]>(vectors.size() * m_dimension);
    m_flannDataset = std::make_shared<flann::Matrix<double>>(
        rawDataPtr.release(), vectors.size(), m_dimension);

    for (size_t i = 0; i < vectors.size(); ++i) {
        for (size_t j = 0; j < m_dimension; ++j) {
            (*m_flannDataset)[i][j] = vectors[i](j);
        }
    }

    // Set up LSH parameters
    flann::LshIndexParams indexParams;
    indexParams.table_number_ = m_numberOfHashTables;
    indexParams.key_size_ = 20;  // You may want to make this configurable
    indexParams.multi_probe_level_ = 2;  // You may want to make this configurable

    // Create and build the LSH index
    m_flannIndex = std::make_unique<flann::Index<flann::L2<double>>>(
        *m_flannDataset, indexParams);
    m_flannIndex->buildIndex();
}

void LSHIndex::addPoint(const VectorXd& vector)
{
    if (!m_flannIndex) {
        throw std::runtime_error("Index not built yet");
    }

    auto newPoint = std::make_unique<flann::Matrix<double>>(
        new double[m_dimension], 1, m_dimension);
    for (size_t i = 0; i < m_dimension; ++i) {
        (*newPoint)[0][i] = vector(i);
    }
    m_flannIndex->addPoints(*newPoint);
}

std::vector<size_t> LSHIndex::findNearestNeighbors(const VectorXd& query, size_t k) const
{
    if (!m_flannIndex) {
        throw std::runtime_error("Index not built yet");
    }

    auto queryPoint = std::make_unique<flann::Matrix<double>>(
        new double[m_dimension], 1, m_dimension);
    for (size_t i = 0; i < m_dimension; ++i) {
        (*queryPoint)[0][i] = query(i);
    }

    std::vector<std::vector<int>> indices;
    std::vector<std::vector<double>> distances;
    flann::SearchParams searchParams(128);  // You may want to make this configurable
    m_flannIndex->knnSearch(*queryPoint, indices, distances, k, searchParams);

    // Convert FLANN indices to our internal indices
    std::vector<size_t> result;
    for (int index : indices[0]) {
        result.push_back(static_cast<size_t>(index));
    }

    return result;
}