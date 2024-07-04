#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <memory>


class VectorDatabase {
    public:
        using VectorXd = Eigen::VectorXd;
        
        enum class IndexType { LSH }; // Later to be implemented KD_TREE, VP_TREE, BALL_TREE, HNSW
        enum class DistanceMetric { EUCLIDEAN, MANHATTAN, COSINE };
        enum class LogLevel { DEBUG, INFO, WARN, ERROR };
        enum class Mode { SPEED, ACCURACY };


        VectorDatabase() = delete; // Default Constructor
        VectorDatabase(const VectorDatabase& other) = delete;         // Copy constructor
        VectorDatabase(VectorDatabase&& other) noexcept; // Move constructor
        VectorDatabase& operator=(const VectorDatabase& other) = delete; // Copy assignment operator
        VectorDatabase& operator=(VectorDatabase&& other) noexcept; // Move assignment operator        
        ~VectorDatabase() = default; // Destructor
        

    public: 
        bool addVector(const Eigen::VectorXd& vec); // Add a vector to the database

        VectorXd getVector(size_t index) const; // Retrieve a vector from the database

        bool serialize(const std::string& filename) const; // Serialize the entire database to a file

        bool deserialize(const std::string& filename); // Deserialize from a file into the database

    private:
    // Helper methods for serialization
        std::string serializeVector(const Eigen::VectorXd& vec) const;
        Eigen::VectorXd deserializeVector(const std::string& data) const;

    private:
        std::vector<VectorXd> m_vectors;
        std::unordered_map<size_t, size_t> m_idToIndexMap;
        size_t m_nextId;
        size_t m_capacity;
        double m_loadFactor;

};