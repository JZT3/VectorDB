#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace flann {
    template <typename T>
    class Index;
    template <typename T>
    struct L2;
    template <typename T>
    class Matrix;
}

class LSHIndex {
public:
    using VectorXd = Eigen::VectorXd;

    LSHIndex(size_t dimension, size_t numberOfHashTables);
    ~LSHIndex() = default;

    // Disable copy operations
    LSHIndex(const LSHIndex&) = delete;
    LSHIndex& operator=(const LSHIndex&) = delete;

    // Enable move operations
    LSHIndex(LSHIndex&&) noexcept = default;
    LSHIndex& operator=(LSHIndex&&) noexcept = default;

    void buildIndex(const std::vector<VectorXd>& vectors);
    void addPoint(const VectorXd& vector);
    std::vector<size_t> findNearestNeighbors(const VectorXd& query, size_t k) const;

private:
    size_t m_dimension;
    size_t m_numberOfHashTables;
    std::unique_ptr<flann::Index<flann::L2<double>>> m_flannIndex;
    std::shared_ptr<flann::Matrix<double>> m_flannDataset;
};