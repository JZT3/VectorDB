#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>

namespace flann {
    template <typename T>
    class Index;
    template <typename T>
    struct L2;
    template <typename T>
    class Matrix;
}

class VectorDatabase {
    public:
        using VectorXd = Eigen::VectorXd;
        
        enum class IndexType { LSH }; // Later to be implemented KD_TREE, VP_TREE, BALL_TREE, HNSW
        enum class DistanceMetric { COSINE }; // Later to be implemented EUCLIDEAN, MANHATTAN, 
        enum class LogLevel { DEBUG, INFO, WARN, ERROR };
        // enum class Mode { SPEED, ACCURACY }; Dynamic modes later to be implemented

        class VectorBuilder 
        {
            public:
                VectorBuilder& setInitialCapacity(size_t capacity) { initialCapacity = capacity; return *this; }
                VectorBuilder& setDimension(size_t dim) { dimension = dim; return *this; }
                VectorBuilder& setDistanceMetric(DistanceMetric metric) { distanceMetric = metric; return *this; }
                VectorBuilder& setStoragePath(const std::string& path) { storagePath = path; return *this; }
                VectorBuilder& setNumberOfHashTables(size_t num) { numberOfHashTables = num; return *this; }
                VectorBuilder& setLSHRadius(double radius) { lshRadius = radius; return *this; }
                //VectorBuilder& setMode(Mode m) { mode = m; return *this; }
                VectorBuilder& setLogLevel(LogLevel level) { logLevel = level; return *this; }
                VectorBuilder& setIndexType(IndexType type) { indexType = type; return *this; }

                VectorDatabase build() const {return VectorDatabase(*this);}

            private:
                friend class VectorDatabase;
                size_t initialCapacity = 1000;
                size_t dimension = 0;
                DistanceMetric distanceMetric = DistanceMetric::COSINE;
                std::string storagePath = "";
                size_t numberOfHashTables = 4;
                double lshRadius = 1.0;
                //Mode mode = Mode::ACCURACY;
                LogLevel logLevel = LogLevel::INFO;
                IndexType indexType = IndexType::LSH;
        };

        VectorDatabase() = delete; // Default Constructor
        VectorDatabase(const VectorDatabase& other) = delete;         // Copy constructor
        VectorDatabase(VectorDatabase&& other) noexcept; // Move constructor
        VectorDatabase& operator=(const VectorDatabase& other) = delete; // Copy assignment operator
        VectorDatabase& operator=(VectorDatabase&& other) noexcept; // Move assignment operator        
        ~VectorDatabase() = default // Destructor
        

    public: 
        bool addVector(const Eigen::VectorXd& vec); // Add a vector to the database
        VectorXd getVector(size_t index) const; // Retrieve a vector from the database
        bool serialize(const std::string& filename) const; // Serialize the entire database to a file
        bool deserialize(const std::string& filename); // Deserialize from a file into the database
        




    private:
    // Helper methods for serialization
        std::string serializeVector(const Eigen::VectorXd& vec) const;
        Eigen::VectorXd deserializeVector(const std::string& data) const;
        std::unique_ptr<flann::Index<flann::L2<double>>> m_flannIndex;
        flann::Matrix<double> m_flannDataset;

    private:
    // Private constructor used by Builder
        explicit VectorDatabase(const VectorBuilder& builder);

    // Helper methods for serialization
        std::string serializeVector(const VectorXd& vec) const;
        VectorXd deserializeVector(const std::string& data) const;
        void initializeDistanceFunction();
        void initializeLSHIndex();

    // Member variables
        std::vector<VectorXd> m_vectors;
        std::unordered_map<size_t, size_t> m_idToIndexMap;
        size_t m_nextId;
        size_t m_capacity;
        size_t m_dimension;
        DistanceMetric m_distanceMetric;
        std::string m_storagePath;
        size_t m_numberOfHashTables;
        double m_lshRadius;
        //Mode m_mode;
        LogLevel m_logLevel;
        IndexType m_indexType;
        
        std::function<double(const VectorXd&, const VectorXd&)> m_distanceFunction;

        std::unique_ptr<flann::Index<flann::L2<double>>> m_flannIndex;
        std::shared_ptr<flann::Matrix<double>> m_flannDataset;
};